"""ImageGen 的业务接口：预设、对象输入、参数优先级及无状态生命周期。"""

import asyncio
import base64
import json
from types import SimpleNamespace
from email import policy
from email.parser import BytesParser
from io import BytesIO

import httpx
from httpx import AsyncClient as HTTPClient
import pytest
from PIL import Image

from llm_config_fixtures import candidate, configuration
from TIYA.llm.image_generator import ImageGen
from TIYA.llm.model import ImagePrompt, Prompts, TextPrompt


@pytest.fixture(autouse=True)
def request_defaults(monkeypatch):
    from TIYA.llm import image_generator

    monkeypatch.setattr(image_generator, "SETTING_CFG", SimpleNamespace(
        LLM=SimpleNamespace(ChatRequestTimeout=300),
    ))


def picture(color="white"):
    output = BytesIO()
    Image.new("RGBA", (16, 16), color).save(output, "PNG")
    return output.getvalue()


def settings(params=None, *, keep_alive=False):
    route = candidate(
        "image_generation:studio:image-test", "image_generation", "studio", "image-test",
        {"endpoint": "https://test/generate", "edit_endpoint": "https://test/edit", "key": "test-key"},
        params=params,
    )
    config = configuration("image-preset", [route])
    config.keep_alive = keep_alive
    config.log_off = True
    return config


def multipart(request):
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + request.headers["content-type"].encode() + b"\r\n\r\n" + request.content
    )
    return [
        (part.get_param("name", header="content-disposition"), part.get_filename(), part.get_payload(decode=True))
        for part in message.iter_parts()
    ]


@pytest.fixture
def service(tmp_path, monkeypatch):
    from TIYA.llm import usage_compat
    from TIYA.llm.usage_compat import LLMUsage

    ledger = LLMUsage(tmp_path / "image-usage.sqlite3")
    monkeypatch.setattr(usage_compat, "_GLOBAL_USAGE", ledger)
    requests = []
    clients = []
    real_client = httpx.AsyncClient

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "data": [{"b64_json": base64.b64encode(picture()).decode()}],
            "usage": {"input_tokens": 20, "output_tokens": 30, "total_tokens": 50},
        })

    def new_client(**kwargs):
        client = real_client(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", new_client)
    return requests, clients, ledger


def test_generation_uses_preset_and_omits_unspecified_parameters(service):
    from TIYA.file_cache import get_file_path_async

    requests, clients, usage = service

    async def run():
        async with ImageGen(config=settings()) as images:
            response = await images.generate(TextPrompt("user", "draw"))
            assert isinstance(response, Prompts)
            assert isinstance(response[0], ImagePrompt)
            assert (await get_file_path_async(response[0].hash_name)).is_file()
            assert response.usage.input_t == 20
            assert response.usage.output_t == 30
            assert response.usage.request_ids

    asyncio.run(run())
    assert json.loads(requests[0].content) == {"model": "image-test", "prompt": "draw"}
    assert all(client.is_closed for client in clients)
    row = usage.ledger.query_requests()[0]
    assert row["identity"]["preset"] == "image-preset"
    assert row["identity"]["provider"] == "image_generation:studio:image-test"


def test_parameters_inherit_and_explicit_defaults_override_without_mutation(service):
    requests, _, usage = service
    config = settings({"quality": "high", "n": 3, "size": None, "stream": True}, keep_alive=True)

    async def run():
        async with ImageGen(config=config) as images:
            await images.generate("inherited", stream=False)
            await images.generate("overridden", quality="auto", n=1, stream=False, output_compression=0)

    asyncio.run(run())
    first, second = [json.loads(request.content) for request in requests]
    assert first["quality"] == "high" and first["n"] == 3
    assert first["stream"] is False and "size" not in first
    assert second["quality"] == "auto" and second["n"] == 1
    assert second["output_compression"] == 0
    assert config.primary.model_params == {"quality": "high", "n": 3, "size": None, "stream": True}
    assert sorted(row["estimated"]["images"] for row in usage.ledger.query_requests()) == [1, 3]


def test_reference_generation_keeps_names_and_order_and_uses_edits(service, tmp_path):
    requests, _, _ = service
    path = tmp_path / "front.png"
    path.write_bytes(picture("red"))
    prompt = ImagePrompt("user", picture("blue"), name="character")

    async def run():
        async with ImageGen(config=settings()) as images:
            await images.generate("reference generation", prompt, path, picture("green"))
        assert prompt.image_input == picture("blue")
        assert not prompt.hash_name

    asyncio.run(run())
    assert requests[0].url.path == "/edit"
    files = [part for part in multipart(requests[0]) if part[0] == "image[]"]
    assert [part[2] for part in files] == [picture("blue"), picture("red"), picture("green")]
    assert files[0][1] == "character.png"
    assert files[1][1] == "front.png"
    assert files[2][1]


def test_edit_base_image_is_first_and_cached_input_is_consumable(service):
    requests, _, _ = service

    async def run():
        async with ImageGen(config=settings()) as images:
            original = await images.generate("first")
            await images.edit("change", original[0], picture("red"), mask=picture())

    asyncio.run(run())
    parts = multipart(requests[1])
    assert [part[2] for part in parts if part[0] == "image[]"] == [picture(), picture("red")]
    assert len([part for part in parts if part[0] == "mask"]) == 1


@pytest.mark.parametrize("name", ["角色/正面", r"角色\正面"])
@pytest.mark.parametrize("source", ["bytes", "path", "base64", "cache"])
def test_reference_alias_is_not_interpreted_as_a_path(service, tmp_path, name, source):
    from TIYA.file_cache import add_file_async

    requests, _, _ = service

    async def run():
        content = picture()
        if source == "path":
            image = tmp_path / "original.png"
            image.write_bytes(content)

        elif source == "base64":
            image = base64.b64encode(content).decode()

        else:
            image = content

        if source == "cache":
            prompt = ImagePrompt("user", name=name, hash_name=await add_file_async(content, "IMAGE"))

        else:
            prompt = ImagePrompt("user", image, name=name)

        async with ImageGen(config=settings()) as images:
            await images.generate("reference", prompt)

        assert prompt.name == name

    asyncio.run(run())
    files = [part for part in multipart(requests[0]) if part[0] == "image[]"]
    assert files[0][1] == f"{name}.png"


def test_markup_is_forwarded_and_mask_markup_conflict_never_sends(service):
    requests, _, _ = service
    transparent = BytesIO()
    Image.new("RGBA", (16, 16), (255, 0, 0, 0)).save(transparent, "PNG")

    async def run():
        async with ImageGen(config=settings()) as images:
            with pytest.raises(ValueError, match="mask.*markup"):
                await images.edit("change", picture(), mask=picture(), markup=transparent.getvalue())
            assert not requests
            await images.edit("change", picture(), markup=transparent.getvalue(), input_fidelity="high")

    asyncio.run(run())
    parts = multipart(requests[0])
    assert [part[2] for part in parts if part[0] == "markup"] == [transparent.getvalue()]
    assert [part[2] for part in parts if part[0] == "input_fidelity"] == [b"high"]


def test_only_first_candidate_is_used_and_errors_are_not_retried(tmp_path, monkeypatch):
    from TIYA.llm import usage_compat
    from TIYA.llm.gateway import HTTPFailure
    from TIYA.llm.usage_compat import LLMUsage

    monkeypatch.setattr(usage_compat, "_GLOBAL_USAGE", LLMUsage(tmp_path / "usage.sqlite3"))
    requests = []
    real_client = httpx.AsyncClient

    def handler(request):
        requests.append(request)
        return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    config = settings()
    config.candidates.append(candidate(
        "image_generation:other:other-model", "image_generation", "other", "other-model",
        {"endpoint": "https://other/generate", "key": "other"},
    ))

    async def run():
        async with ImageGen(config=config) as images:
            with pytest.raises(HTTPFailure):
                await images.generate("draw")

    asyncio.run(run())
    assert len(requests) == 1
    assert json.loads(requests[0].content)["model"] == "image-test"
    assert requests[0].url.host == "test"


@pytest.mark.parametrize("keep_alive", [False, True])
def test_connection_lifecycle_and_closed_client(service, keep_alive):
    _, clients, _ = service

    async def run():
        async with ImageGen(config=settings(keep_alive=keep_alive)) as images:
            await images.generate("first")
            assert clients[0].is_closed is not keep_alive
            await images.generate("second")
            assert len(clients) == (1 if keep_alive else 2)

        assert all(client.is_closed for client in clients)
        await images.close()
        with pytest.raises(RuntimeError, match="关闭"):
            await images.generate("after close")

    asyncio.run(run())


def test_additional_usage_ledger_gets_the_same_request(service, tmp_path):
    from TIYA.llm.usage_compat import LLMUsage

    _, _, primary = service
    extra = LLMUsage(tmp_path / "extra.sqlite3")

    async def run():
        async with ImageGen(config=settings(), usage_record=[extra]) as images:
            await images.generate("draw")

    asyncio.run(run())
    assert extra.ledger.query_requests()[0]["id"] == primary.ledger.query_requests()[0]["id"]


def test_direct_settings_forward_extensions_and_request_timeout_is_not_payload(service):
    requests, _, _ = service

    async def run():
        async with ImageGen(endpoint="https://test/generate", token="k", model="direct", log_off=True) as images:
            await images.generate("draw", complexity="high", frame_count=4, frame_duration_ms=200, timeout=5)

    asyncio.run(run())
    assert json.loads(requests[0].content) == {
        "model": "direct", "prompt": "draw", "complexity": "high", "frame_count": 4, "frame_duration_ms": 200,
        "stream": False,
    }


def test_base64_input_and_object_parameters_are_used_on_later_calls(service):
    requests, _, _ = service
    encoded = ImagePrompt("user", base64.b64encode(picture()).decode(), name="reference")

    async def run():
        async with ImageGen(config=settings()) as images:
            await images.generate("first", encoded)
            images.model_params.update(quality="low", seed=7, unused=None)
            await images.generate("second", seed=9, quality=None)

    asyncio.run(run())
    assert [part[2] for part in multipart(requests[0]) if part[0] == "image[]"] == [picture()]
    assert json.loads(requests[1].content) == {
        "model": "image-test", "prompt": "second", "quality": "low", "seed": 9,
    }


def test_stream_returns_only_final_cached_images(service, monkeypatch):
    requests, clients, usage = service
    events = [
        {"type": "image_generation.partial_image", "b64_json": base64.b64encode(picture("red")).decode()},
        {
            "type": "image_generation.completed",
            "b64_json": base64.b64encode(picture("blue")).decode(),
            "usage": {"input_tokens": 10, "output_tokens": 15},
        },
    ]

    def handler(request):
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content="".join(
            f"data: {json.dumps(event)}\n\n" for event in events
        ))

    from TIYA.file_cache import get_file_path_async

    def new_client(**kwargs):
        client = HTTPClient(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", new_client)

    async def run():
        async with ImageGen(config=settings()) as images:
            result = await images.generate("stream", stream=True)
            assert len(list(result)) == 1
            assert (await get_file_path_async(result[0].hash_name)).read_bytes() == picture("blue")
            assert result.usage.input_t == 10

    asyncio.run(run())
    assert len(requests) == 1
    assert usage.ledger.query_requests()[0]["reported"]["images"] == 1


@pytest.mark.parametrize("stop", ["timeout", "cancel"])
def test_interrupted_request_is_not_retried_and_connection_closes(service, monkeypatch, stop):
    _, clients, usage = service
    from TIYA.llm.gateway import Gateway

    async def run():
        entered = asyncio.Event()
        sent = 0

        async def wait_for_response(self, request, timeout):
            nonlocal sent
            sent += 1
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(Gateway, "send", wait_for_response)
        async with ImageGen(config=settings()) as images:
            if stop == "timeout":
                with pytest.raises(TimeoutError):
                    await images.generate("draw", timeout=1)

            else:
                task = asyncio.create_task(images.generate("draw"))
                await asyncio.wait_for(entered.wait(), 2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

        assert sent == 1

    asyncio.run(run())
    assert clients and all(client.is_closed for client in clients)
    assert len(usage.ledger.query_requests()) == 1
    assert usage.ledger.query_requests()[0]["status"] in {"unknown", "cancelled"}


def test_concurrent_calls_are_independent_without_queue(service, monkeypatch):
    requests, clients, _ = service
    from TIYA.llm.gateway import Gateway
    original = Gateway.send

    async def run():
        both_entered = asyncio.Event()
        entered = 0

        async def send(self, request, timeout):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()

            await asyncio.wait_for(both_entered.wait(), 2)
            return await original(self, request, timeout)

        monkeypatch.setattr(Gateway, "send", send)
        async with ImageGen(config=settings()) as images:
            first, second = await asyncio.gather(images.generate("first"), images.generate("second"))
            assert first.usage.request_ids != second.usage.request_ids

    asyncio.run(run())
    assert len(clients) == 2 and all(client.is_closed for client in clients)
    assert {json.loads(request.content)["prompt"] for request in requests} == {"first", "second"}


def test_invalid_local_inputs_do_not_send_requests(service):
    requests, _, _ = service

    async def run():
        async with ImageGen(config=settings()) as images:
            with pytest.raises(TypeError, match="prompt"):
                await images.generate(object())

            with pytest.raises(ValueError, match="图片"):
                await images.generate("draw", ImagePrompt("user"))

            with pytest.raises(ValueError, match="timeout"):
                await images.generate("draw", timeout=float("inf"))

    asyncio.run(run())
    assert not requests
