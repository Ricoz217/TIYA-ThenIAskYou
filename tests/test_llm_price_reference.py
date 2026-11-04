"""配置引用就是价格查询键；预设名和凭证不参与价格关联。"""

import asyncio
from decimal import Decimal

import httpx
import pytest

from llm_config_fixtures import candidate, configuration
from TIYA.llm.chat import Chat
from TIYA.llm.pricing import PriceCatalog
from TIYA.llm.runtime import Runtime
from TIYA.llm.state import StateStore
from TIYA.llm.usage import Ledger
from TIYA.llm.usage_summary import MonthlyOverview


REF = "chat_completions:deepseek:deepseek-flash"
RULES = {
    "currency": "CNY",
    "units": {"input_uncached": 1_000_000, "cached": 1_000_000, "output": 1_000_000},
    "tiers": [{"rates": {"input_uncached": "2", "cached": "0.04", "output": "8"}}],
}


def test_reference_lookup_and_frozen_price_survive_new_version(tmp_path):
    prices = PriceCatalog(tmp_path / "prices.db")
    first = prices.publish(REF, RULES, source="peak", effective=1)
    frozen = prices.snapshot(REF, at=2)
    assert frozen["ref"] == REF
    assert frozen["version"] == first
    prices.publish(REF, {**RULES, "currency": "USD"}, source="revised", effective=3)
    assert prices.snapshot(REF, at=4)["rules"]["currency"] == "USD"
    assert frozen["rules"]["currency"] == "CNY"
    assert prices.snapshot("chat_completions:other:deepseek-flash") is None
    assert prices.snapshot("responses_api:deepseek:deepseek-flash") is None


@pytest.mark.parametrize("ref", ["", "deepseek-flash", "chat_completions:deepseek", "a::m", "a:b:"])
def test_price_reference_requires_complete_config_id(tmp_path, ref):
    prices = PriceCatalog(tmp_path / "prices.db")
    with pytest.raises(ValueError):
        prices.publish(ref, RULES, source="invalid")


def test_monthly_missing_price_uses_provider_ref_across_presets_and_scopes(tmp_path):
    prices = PriceCatalog(tmp_path / "prices.db")
    ledger = Ledger(tmp_path / "usage.db")
    records = []
    for preset, scope in [("MAIN", "old-key"), ("MEMORY", "new-key")]:
        request_id = ledger.begin_request(
            ledger.begin_call(preset),
            {"preset": preset, "scope": scope, "provider": REF, "model": "deepseek-flash", "operation": "chat_completions"},
            {"input": 100, "cached": 20, "output": 10},
        )
        records.append(ledger.get_request(request_id))

    view = MonthlyOverview(ledger, prices)
    assert view.summary()["unknown_cost_requests"] == 2
    prices.publish(REF, RULES, source="peak")
    assert view.summary()["cost"]["CNY"] == pytest.approx(0.0004816)
    assert view.summary()["unknown_cost_requests"] == 0
    assert [ledger.get_request(row["id"]) for row in records] == records


@pytest.mark.parametrize("category,connection,model,currency", [
    ("chat_completions", "deepseek", "deepseek-flash", "CNY"),
    ("chat_completions", "openai", "gpt-6.1-sol", "USD"),
    ("responses_api", "openai", "gpt-6.1-sol", "USD"),
])
def test_chat_automatically_prices_reference_independently_of_preset_and_key(
    tmp_path,
    monkeypatch,
    category,
    connection,
    model,
    currency,
):
    real_client = httpx.AsyncClient
    ref = f"{category}:{connection}:{model}"

    def handler(request):
        if category == "responses_api":
            return httpx.Response(200, json={
                "id": "response-test", "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 100, "output_tokens": 10, "input_tokens_details": {"cached_tokens": 20}},
            })

        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 20}},
        })

    def client_factory(**kwargs):
        kwargs.pop("mounts", None)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    prices = PriceCatalog(tmp_path / "prices.db")
    version = prices.publish(ref, {**RULES, "currency": currency}, source="test")
    ledger = Ledger(tmp_path / "usage.db")

    async def run():
        for name, key in [("MAIN", "key-one"), ("AGENT", "key-two")]:
            route = candidate(
                ref, category, connection, model,
                {"endpoint": "https://test/chat/completions", "key": key}, {"stream": False},
            )
            chat = Chat(config=configuration(name, [route]), log_off=True)
            chat._runtime = Runtime(StateStore(tmp_path / "state.db"), ledger, prices)
            try:
                assert (await chat.ask("hello", timeout=5))[0].text == "ok"

            finally:
                await chat.close()

    asyncio.run(run())
    rows = ledger.query_requests()
    assert len(rows) == 2
    assert rows[0]["identity"]["scope"] != rows[1]["identity"]["scope"]
    for row in rows:
        assert row["price_snapshot"]["version"] == version
        assert row["price_snapshot"]["ref"] == ref
        assert row["charge"]["currency"] == currency
        assert Decimal(row["charge"]["amount"]) == Decimal("0.0002408")


def test_image_client_prices_same_reference_without_operation_binding(tmp_path):
    import base64
    from io import BytesIO
    from PIL import Image
    from TIYA.llm.clients import ImagesClient
    from TIYA.llm.model import ImageGenerateRequest

    model = "gpt-image-2.5-sunburst"
    ref = f"image_generation:openai:{model}"
    prices = PriceCatalog(tmp_path / "prices.db")
    version = prices.publish(
        ref,
        {"currency": "USD", "threshold": "images", "tiers": [{"rates": {"images": "2"}}]},
        source="test",
    )
    ledger = Ledger(tmp_path / "usage.db")
    content = BytesIO()
    Image.new("RGB", (1, 1)).save(content, "PNG")

    def handler(request):
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(content.getvalue()).decode()}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            images = ImagesClient(
                {"endpoint": "https://test/generate"},
                provider_ref="image_generation:openai",
                client=client,
                ledger=ledger,
                state=StateStore(tmp_path / "state.db"),
                prices=prices,
            )
            assert images.pricing_identity(model) == ref
            await images.generate(ImageGenerateRequest(model, "draw"), preset="ANY")
            await images.close()

    asyncio.run(run())
    row = ledger.query_requests()[0]
    assert row["identity"]["provider"] == ref
    assert row["price_snapshot"]["version"] == version
    assert Decimal(row["charge"]["amount"]) == Decimal(2)
