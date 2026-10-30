from TIYA.LLM_usage import ApiPrice, LLMUsage, UsageStorage


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

    today = next(
        iter(next(iter(next(iter(usage._storage.values())).values())).values())
    )

    assert len(today) == 2
    assert {item.price is None for item in today.values()} == {True, False}
