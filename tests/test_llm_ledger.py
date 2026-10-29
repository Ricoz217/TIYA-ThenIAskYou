from decimal import Decimal

import pytest


def test_partial_reported_values_replace_estimates_per_field_and_survive_restart(
    tmp_path,
):
    from TIYA.llm.usage import Ledger

    path = tmp_path / "usage.sqlite3"
    ledger = Ledger(path)
    call = ledger.begin_call("MAIN")
    request = ledger.begin_request(
        call,
        {"model": "m", "operation": "chat"},
        {"input": 100, "output": 5, "cached": 0},
    )
    observation = ledger.observe_usage(
        request, "reported", {"input": 90, "cached": 30}, "final"
    )
    assert (
        ledger.observe_usage(request, "reported", {"input": 90, "cached": 30}, "final")
        == observation
    )
    ledger.finish_request(request, "completed")
    row = Ledger(path).get_request(request)
    assert row["effective"] == {"input": 90, "output": 5, "cached": 30}
    assert row["estimated"]["input"] == 100
    assert row["reported"].get("output") is None
    assert len(ledger.query_requests()) == 1
    assert ledger.summarize()["input"]["verified"] == 90


def test_missing_and_explicit_zero_prices_and_tiers(tmp_path):
    from TIYA.llm.pricing import PriceCatalog

    prices = PriceCatalog(tmp_path / "pricing.sqlite3")
    identity = {"preset": "MAIN", "scope": "account", "model": "m", "operation": "chat"}
    rules = {
        "currency": "CNY",
        "tier_mode": "whole",
        "threshold": "input",
        "tiers": [
            {
                "up_to": 100,
                "rates": {"input_uncached": "1", "cached": "0", "output": "2"},
            },
            {"rates": {"input_uncached": "3", "cached": "0", "output": "4"}},
        ],
        "units": {"input_uncached": 1000000, "cached": 1000000, "output": 1000000},
    }
    prices.publish(identity, rules, source="test")
    snapshot = prices.snapshot(identity)
    charge = prices.calculate(snapshot, {"input": 200, "cached": 50, "output": 10})
    assert charge["amount"] == Decimal("0.00049")
    prices.publish(identity, {**rules, "currency": "USD"}, source="revision")
    assert (
        prices.calculate(snapshot, {"input": 200, "cached": 50, "output": 10})[
            "currency"
        ]
        == "CNY"
    )
    assert prices.snapshot({**identity, "scope": "different"}) is None
    assert prices.calculate(None, {"input": 200})["amount"] is None


def test_failed_attempts_have_distinct_ids_and_unknown_output(tmp_path):
    from TIYA.llm.usage import Ledger

    ledger = Ledger(tmp_path / "ledger.sqlite3")
    call = ledger.begin_call("p")
    first = ledger.begin_request(call, {"model": "m"}, {"input": 10})
    ledger.finish_request(first, "unknown", remote_state="unknown")
    second = ledger.begin_request(call, {"model": "m"}, {"input": 10})
    assert first != second
    assert ledger.get_request(first)["effective"].get("output") is None
    assert {r["call_id"] for r in ledger.query_requests()} == {call}


def test_import_aggregate_is_idempotent_and_does_not_import_last_record_twice(tmp_path):
    from TIYA.llm.usage import Ledger

    ledger = Ledger(tmp_path / "ledger.sqlite3")
    old = {
        "storage": {
            "2026": {
                "10": {
                    "29": {
                        "hash": {
                            "model": "m",
                            "price": None,
                            "usage": {
                                "input": {"verified": 100, "estimate": 5},
                                "output": {"verified": 10, "estimate": 0},
                                "cached": {"verified": 90, "estimate": 0},
                            },
                        }
                    }
                }
            }
        },
        "last_record": {"ignored": {"model": "m"}},
    }
    ledger.import_aggregate(old, "old")
    ledger.import_aggregate(old, "old")
    assert ledger.summarize()["input"] == {"verified": 100, "estimate": 5}
    assert ledger.query_requests() == []


def test_queries_filter_identity_and_keep_initial_estimate_history(tmp_path):
    from TIYA.llm.usage import Ledger

    ledger = Ledger(tmp_path / "ledger.sqlite3")
    call = ledger.begin_call("p")
    request = ledger.begin_request(
        call, {"preset": "p", "model": "m", "operation": "chat"}, {"input": 20}
    )
    ledger.observe_usage(request, "reported", {"input": 10, "cached": 5}, "final")
    ledger.finish_request(request, "completed")
    assert (
        len(
            ledger.query_requests(
                preset="p", model="m", operation="chat", status="completed"
            )
        )
        == 1
    )
    assert ledger.query_requests(model="different") == []
    observations = ledger.query_observations(request)
    assert observations[0]["source"] == "estimated"
    assert observations[0]["values"]["input"] == 20
    assert observations[1]["source"] == "reported"
    assert ledger.summarize()["cache_basis"] == {"input": 10, "cached": 5}


def test_standalone_image_price_and_explicit_free_plan_need_no_fake_preset(tmp_path):
    from TIYA.llm.pricing import PriceCatalog

    catalog = PriceCatalog(tmp_path / "prices.sqlite3")
    identity = {
        "preset": "",
        "scope": "account",
        "model": "image-2.5",
        "operation": "images_generate",
    }
    catalog.publish(
        identity,
        {"currency": "CNY", "tiers": [{"rates": {"input": "0", "output": "0"}}]},
        source="explicit-free",
    )
    assert catalog.calculate(catalog.snapshot(identity), {})["amount"] == Decimal(0)
