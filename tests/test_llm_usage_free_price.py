from TIYA import LLM_connect
from TIYA.LLM_usage import ApiPrice, LLMUsage, UsageStorage
from TIYA.utils import DecoratedDict


def _llm_config(price: dict) -> DecoratedDict:
    return DecoratedDict(
        {
            "endpoint": "https://example.com/v1/chat/completions",
            "token": "test-token",
            "model": "free-vision-model",
            "api_type": "openai",
            "proxy_mode": "",
            "max_context": 128_000,
            "auto_compress_rate": 0.7,
            "price": price,
            "extra_parameter": {},
        }
    )


def test_zero_price_and_missing_price_have_different_hashes():
    zero_price = ApiPrice(input=0, output=0, cached=0, currency="CNY")

    missing_price_usage = UsageStorage("free-vision-model", None)
    zero_price_usage = UsageStorage("free-vision-model", zero_price)

    assert missing_price_usage.hash_id() != zero_price_usage.hash_id()
    assert zero_price_usage.to_dict()["price"] == zero_price.to_dict()


def test_usage_storage_keeps_missing_and_zero_price_separate(tmp_path):
    usage = LLMUsage(tmp_path / "usage.json")

    usage.update("free-vision-model", {"input": 10, "output": 2}, price=None)
    usage.update(
        "free-vision-model",
        {"input": 20, "output": 4},
        price=ApiPrice(input=0, output=0, cached=0, currency="CNY"),
    )

    today = next(iter(next(iter(next(iter(usage._storage.values())).values())).values()))

    assert len(today) == 2
    assert {item.price is None for item in today.values()} == {True, False}


def test_parse_llm_setting_preserves_explicit_zero_price(monkeypatch):
    monkeypatch.setattr(
        LLM_connect,
        "get_llm",
        lambda _: _llm_config(
            {
                "currency": "CNY",
                "input_token": 0,
                "cache_hit": 0,
                "output_token": 0,
            }
        ),
    )

    setting = LLM_connect.parse_llm_setting("FREE_VISION")

    assert setting.price == ApiPrice(input=0, output=0, cached=0, currency="CNY")


def test_parse_llm_setting_keeps_empty_price_unconfigured(monkeypatch):
    monkeypatch.setattr(LLM_connect, "get_llm", lambda _: _llm_config({}))

    setting = LLM_connect.parse_llm_setting("NO_PRICE")

    assert setting.price is None
