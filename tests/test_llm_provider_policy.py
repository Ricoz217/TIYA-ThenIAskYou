import asyncio
import json

import httpx
import pytest

from llm_config_fixtures import candidate, configuration
from TIYA.llm.gateway import Gateway, HTTPFailure
from TIYA.llm.model import Context, ImagePrompt, Prompts, TextPrompt
from TIYA.llm.provider import REGISTRY
from TIYA.llm.provider.base import Failure, ProviderFailure, classify_error
from TIYA.llm.provider.openai_chat import ChatCompletions
from TIYA.llm.provider.openai_responses import Responses
from TIYA.llm.runtime import Runtime
from TIYA.llm.state import StateStore
from TIYA.llm.usage import Ledger
from TIYA.utils import fingerprint


@pytest.mark.parametrize("status,code,retryable,fallback,recovery", [
    (429, "rate_limit_exceeded", True, True, "none"),
    (429, "insufficient_quota", False, True, "none"),
    (401, "invalid_api_key", False, True, "none"),
    (400, "invalid_request_error", False, True, "none"),
    (400, "context_length_exceeded", False, False, "context_overflow"),
    (400, "invalid_previous_response_id", False, True, "rebuild_parent"),
    (400, "file_expired", False, True, "refresh_file"),
    (400, "unsupported_file_reference", False, True, "disable_files"),
    (503, "server_error", False, False, "none"),
])
def test_provider_classification_explicitly_describes_allowed_actions(status, code, retryable, fallback, recovery):
    failure = classify_error(status, {"error": {"code": code}})
    assert failure.retryable is retryable
    assert failure.fallback_allowed is fallback
    assert failure.recovery == recovery


def services(tmp_path):
    return Runtime(StateStore(tmp_path / "state.db"), Ledger(tmp_path / "usage.db"))


def config_for(category):
    return configuration("main", [
        candidate(f"{category}:a:m", category, "a", "m", {"endpoint": "https://a/chat"}, {"stream": False}),
        candidate("chat_completions:b:m", "chat_completions", "b", "m", {"endpoint": "https://b/chat"}, {"stream": False}),
    ])


@pytest.mark.parametrize("retryable,fallback,recovery,expected", [
    (True, False, "none", ["a", "a"]),
    (True, False, "refresh_file", ["a", "a"]),
    (False, True, "none", ["a", "b"]),
    (False, False, "none", ["a"]),
])
def test_runtime_obeys_provider_flags_instead_of_category(tmp_path, monkeypatch, retryable, fallback, recovery, expected):
    class Custom(ChatCompletions):
        def failure(self, status, body, headers=None):
            return Failure(
                "vendor_specific", remote_state="rejected",
                retryable=retryable, fallback_allowed=fallback,
                recovery=recovery,
            )

    monkeypatch.setitem(REGISTRY, "policy", Custom)
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})

        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def run():
        runtime = services(tmp_path)
        config = config_for("policy")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            if not retryable and not fallback:
                with pytest.raises(HTTPFailure):
                    await runtime.execute(Context(), Prompts(TextPrompt("user", "hi")), config, gateway, 5)

            else:
                pending = await runtime.execute(Context(), Prompts(TextPrompt("user", "hi")), config, gateway, 5)
                assert pending.result.prompts[0].text == "ok"

    asyncio.run(run())
    assert calls == expected


@pytest.mark.parametrize("attempts", [1, 6])
def test_retry_limit_does_not_implicitly_allow_provider_forbidden_fallback(tmp_path, monkeypatch, attempts):
    class Custom(ChatCompletions):
        def failure(self, status, body, headers=None):
            return Failure("rate_limit", remote_state="rejected", retryable=True, fallback_allowed=False)

    monkeypatch.setitem(REGISTRY, "retry_only", Custom)
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})

    async def run():
        runtime = services(tmp_path)
        config = config_for("retry_only")
        config.request_policy.max_attempts = attempts
        config.refresh_fingerprints()
        context = Context()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            with pytest.raises(HTTPFailure):
                await runtime.execute(context, Prompts(TextPrompt("user", "hi")), config, gateway, 5)

            assert runtime.state.get("cooldown", fingerprint([context.session_id, config.revision])) is None

    asyncio.run(run())
    assert calls == ["a"] * min(attempts, 2)


def test_provider_can_disable_retry_and_fallback_for_connection_errors(tmp_path, monkeypatch):
    class Custom(ChatCompletions):
        def transport_failure(self, error):
            return Failure("connection", remote_state="rejected", retryable=False, fallback_allowed=False)

    monkeypatch.setitem(REGISTRY, "stop_transport", Custom)
    calls = []

    def handler(request):
        calls.append(request.url.host)
        raise httpx.ConnectError("connection refused")

    async def run():
        runtime = services(tmp_path)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            with pytest.raises(httpx.ConnectError):
                await runtime.execute(Context(), Prompts(TextPrompt("user", "hi")), config_for("stop_transport"), gateway, 5)

    asyncio.run(run())
    assert calls == ["a"]


@pytest.mark.parametrize("evidence", ["partial", "usage", "body_usage", "stream_usage", "unknown"])
def test_runtime_execution_evidence_vetoes_provider_resend_permission(tmp_path, monkeypatch, evidence):
    recommendations = []

    class Custom(ChatCompletions):
        def failure(self, status, body, headers=None):
            failure = Failure(
                "vendor_specific", remote_state="unknown" if evidence == "unknown" else "rejected",
                retryable=True, fallback_allowed=True,
            )
            recommendations.append(failure)
            return failure

    monkeypatch.setitem(REGISTRY, "unsafe", Custom)
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if evidence in ("partial", "stream_usage"):
            first = (
                '{"choices":[{"delta":{"content":"started"}}]}'
                if evidence == "partial"
                else '{"choices":[],"usage":{"prompt_tokens":10}}'
            )
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=(
                f'data: {first}\n\n'
                'data: {"error":{"code":"vendor_specific"}}\n\n'
            ))

        body = {"error": {"code": "vendor_specific"}}
        if evidence in ("usage", "body_usage"):
            body["usage"] = {"prompt_tokens": 10}

        return httpx.Response(200 if evidence == "body_usage" else 429, json=body)

    async def run():
        runtime = services(tmp_path)
        config = config_for("unsafe")
        config.primary.model_params["stream"] = evidence in ("partial", "stream_usage")
        config.refresh_fingerprints()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            with pytest.raises((HTTPFailure, ProviderFailure)) as caught:
                await runtime.execute(Context(), Prompts(TextPrompt("user", "hi")), config, gateway, 5)

            assert caught.value.llm_outcome.remote_state == "unknown"
            row = runtime.ledger.query_requests()[0]
            assert row["diagnostic"]["retryable"] is True
            assert row["diagnostic"]["fallback_allowed"] is True
            assert row["diagnostic"]["resend_safe"] is False
            if evidence in ("usage", "body_usage", "stream_usage"):
                assert row["reported"]["input"] == 10

    asyncio.run(run())
    assert calls == ["a"]
    assert recommendations[0].remote_state == ("unknown" if evidence == "unknown" else "rejected")


def test_provider_recovery_action_is_independent_of_diagnostic_category(tmp_path, monkeypatch):
    class Custom(Responses):
        def failure(self, status, body, headers=None):
            return Failure(
                "vendor_history_gone", remote_state="rejected",
                recovery="rebuild_parent", fallback_allowed=False,
            )

    monkeypatch.setitem(REGISTRY, "responses_api", Custom)
    config = configuration("main", [candidate(
        "responses_api:a:m", "responses_api", "a", "m",
        {"endpoint": "https://a/responses", "responses": {"supports_previous_response_id": True}},
        {"stream": False},
    )])
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("previous_response_id"):
            return httpx.Response(400, json={"error": {"code": "vendor_history_gone"}})

        return httpx.Response(200, json={"id": f"r{len(bodies)}", "output": []})

    async def run():
        runtime = services(tmp_path)
        context = Context()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            first = await runtime.execute(context, Prompts(TextPrompt("user", "one")), config, gateway, 5)
            runtime.commit(context, first)
            await runtime.execute(context, Prompts(TextPrompt("user", "two")), config, gateway, 5)

    asyncio.run(run())
    assert len(bodies) == 3
    assert bodies[1]["previous_response_id"] == "r1"
    assert "previous_response_id" not in bodies[2]
    assert len(bodies[2]["input"]) == 2


def test_media_preflight_obeys_provider_forbidden_fallback(tmp_path, monkeypatch):
    runtime = services(tmp_path)
    prepared_routes = []

    async def reject_image(*args, **kwargs):
        raise ProviderFailure(Failure("image_rejected", remote_state="rejected", fallback_allowed=False))

    monkeypatch.setattr(runtime.media, "prepare_prompt", reject_image)

    async def gateway(route):
        prepared_routes.append(route.connection_name)
        return None  # 图片预检失败，不会进入网络发送。

    with pytest.raises(ProviderFailure):
        asyncio.run(runtime.execute(Context(), Prompts(ImagePrompt("user", b"image")),
                                    config_for("chat_completions"), gateway, 5))

    assert prepared_routes == ["a"]
    assert runtime.ledger.query_requests() == []


def test_completed_result_vetoes_provider_permission_to_resend_local_failure(tmp_path, monkeypatch):
    class Custom(ChatCompletions):
        def transport_failure(self, error):
            return Failure("local_failure", remote_state="rejected", retryable=True, fallback_allowed=True)

    monkeypatch.setitem(REGISTRY, "completed", Custom)
    runtime = services(tmp_path)
    calls = []

    async def fail_local_commit(*args):
        raise OSError("local result processing failed")

    monkeypatch.setattr(runtime, "_complete_attempt", fail_local_commit)

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            with pytest.raises(OSError):
                await runtime.execute(Context(), Prompts(TextPrompt("user", "hi")), config_for("completed"), gateway, 5)

            row = runtime.ledger.query_requests()[0]
            assert row["remote_state"] == "completed"
            assert row["diagnostic"]["resend_safe"] is False
            assert row["estimated"]["input"] > 0

    asyncio.run(run())
    assert calls == ["a"]


@pytest.mark.parametrize("strict", [False, True])
def test_files_provider_controls_upload_to_inline_fallback(tmp_path, monkeypatch, strict):
    from PIL import Image
    from TIYA.llm import media as module
    from TIYA.llm.media import MediaService
    from TIYA.llm.provider.files import FilesProvider, REGISTRY as FILES
    from TIYA.llm.provider import get_provider

    class Custom(FilesProvider):
        def failure(self, status, body, headers=None):
            failure = super().failure(status, body, headers)
            if strict:
                failure.fallback_allowed = False
                failure.recovery = "none"

            return failure

    monkeypatch.setitem(FILES, "files_policy", Custom)
    path = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(path)

    async def add(*args):
        return "cached"

    async def get(*args):
        return path

    monkeypatch.setattr(module, "add_file_async", add)
    monkeypatch.setattr(module, "get_file_path_async", get)
    uploads = []

    async def send(wire):
        uploads.append(wire)
        raise HTTPFailure(503, {"error": {"code": "server_error"}})

    route = candidate(
        "responses_api:a:m", "responses_api", "a", "m", {"endpoint": "https://a/responses"},
        files_ref="files_policy:files", files_settings={"endpoint": "https://files/upload"},
    )
    service = MediaService(StateStore(tmp_path / "state.db"))

    async def run():
        prompt = ImagePrompt("user", path)
        if strict:
            with pytest.raises(ProviderFailure):
                await service.prepare_prompt(prompt, get_provider(route), send)

        else:
            refs = await service.prepare_prompt(prompt, get_provider(route), send)
            assert refs[0].kind == "base64"

    asyncio.run(run())
    assert len(uploads) == 1


@pytest.mark.parametrize("value,expected", [("0", 0), ("-1", None), ("nan", None), ("inf", None)])
def test_provider_retry_delay_is_zero_or_a_finite_nonnegative_number(value, expected):
    failure = classify_error(429, {}, headers={"retry-after": value})
    assert failure.retry_after == expected


def test_error_resource_is_optional_string_and_unknown_extra_fields_are_ignored():
    failure = classify_error(400, {
        "error": {"code": "file_expired", "file_id": ["invalid"]},
        "base_resp": None,
    })
    assert failure.resource is None
    raw_error = failure.raw["error"]
    assert isinstance(raw_error, dict)
    assert raw_error["file_id"] == ["invalid"]

    failure = classify_error(200, {"base_resp": None}, "minimax")
    assert failure.category == "unknown"
    assert failure.retryable is False


def test_non_object_http_error_still_finishes_accounting_without_fallback(tmp_path):
    runtime = services(tmp_path)
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(503, json=["unexpected error shape"])

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            with pytest.raises(HTTPFailure):
                await runtime.execute(Context(), Prompts(TextPrompt("user", "hi")),
                                      config_for("chat_completions"), gateway, 5)

            assert runtime.ledger.query_requests()[0]["status"] == "unknown"

    asyncio.run(run())
    assert calls == ["a"]


def test_background_poll_failure_never_creates_a_second_generation(tmp_path):
    runtime = services(tmp_path)
    config = configuration("main", [candidate(
        "responses_api:a:m", "responses_api", "a", "m",
        {"endpoint": "https://a/responses", "responses": {"supports_background": True}},
        {"stream": False, "background": True},
    )])
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(200, json={"id": "created", "status": "queued", "output": []})

        return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            with pytest.raises(ProviderFailure) as caught:
                await runtime.execute(Context(), Prompts(TextPrompt("user", "hi")), config, gateway, 5)

            assert caught.value.llm_outcome.remote_state == "unknown"
            rows = runtime.ledger.query_requests()
            assert len(rows) == 2

    asyncio.run(run())
    assert requests == [("POST", "/responses"), ("GET", "/responses/created")]
