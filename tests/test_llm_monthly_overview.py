"""月度状态栏的临时估价与增量刷新，不修改请求账本。"""

from datetime import datetime
from decimal import Decimal

import pytest

from TIYA.llm.pricing import PriceCatalog
from TIYA.llm.usage import Ledger


REF = "chat_completions:account:m"
IDENTITY = {
    "preset": "MAIN",
    "provider": REF,
    "scope": "account",
    "model": "m",
    "operation": "chat_completions",
}


def publish(catalog, rate="2", **kwargs):
    return catalog.publish(
        REF,
        {
            "currency": "CNY",
            "threshold": "input",
            "units": {"input_uncached": 100, "cached": 100, "output": 100},
            "tiers": [{"rates": {"input_uncached": rate, "cached": "0.5", "output": "4"}}],
        },
        source="test",
        **kwargs,
    )


def request(ledger, *, snapshot=None, identity=None, input_tokens=100):
    call = ledger.begin_call("MAIN")
    request_id = ledger.begin_request(
        call,
        identity or IDENTITY,
        {"input": input_tokens, "cached": 20, "output": 10},
        price_snapshot=snapshot,
    )
    ledger.observe_usage(
        request_id, "reported", {"input": input_tokens, "cached": 20, "output": 10}, "final"
    )
    ledger.finish_request(request_id, "completed")
    return request_id


def test_new_price_revalues_only_display_and_never_writes_old_records(tmp_path):
    from TIYA.llm.usage_summary import MonthlyOverview

    ledger = Ledger(tmp_path / "usage.sqlite3")
    prices = PriceCatalog(tmp_path / "prices.sqlite3")
    request_id = request(ledger)
    original = ledger.get_request(request_id)
    view = MonthlyOverview(ledger, prices)
    assert view.summary()["unknown_cost_requests"] == 1

    publish(prices)
    result = view.summary()
    assert result["cost"]["CNY"] == pytest.approx(2.1)
    assert result["temporary_cost"] == {"CNY": pytest.approx(2.1)}
    assert result["unknown_cost_requests"] == 0
    assert ledger.get_request(request_id) == original
    assert ledger.summarize()["unknown_cost_requests"] == 1
    with ledger.db.transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM charges").fetchone()[0] == 1


def test_frozen_charges_stay_frozen_and_unknown_prices_follow_current_catalog(tmp_path):
    from TIYA.llm.usage_summary import MonthlyOverview

    ledger = Ledger(tmp_path / "usage.sqlite3")
    prices = PriceCatalog(tmp_path / "prices.sqlite3")
    unpriced = request(ledger)
    publish(prices)
    priced = request(ledger, snapshot=prices.snapshot(REF))
    view = MonthlyOverview(ledger, prices)
    assert view.summary()["cost"]["CNY"] == pytest.approx(4.2)

    publish(prices, "3")
    assert view.summary()["cost"]["CNY"] == pytest.approx(5.0)
    assert view.summary()["temporary_cost"]["CNY"] == pytest.approx(2.9)
    assert ledger.get_request(unpriced)["price_snapshot"] is None
    assert Decimal(ledger.get_request(priced)["charge"]["amount"]) == Decimal("2.1")


def test_incremental_updates_replacements_and_reopened_writer_do_not_double_count(tmp_path):
    from TIYA.llm.usage_summary import MonthlyOverview

    ledger = Ledger(tmp_path / "usage.sqlite3")
    prices = PriceCatalog(tmp_path / "prices.sqlite3")
    publish(prices)
    request_id = request(ledger)
    view = MonthlyOverview(ledger, prices)
    assert view.summary()["input"] == {"verified": 100, "estimate": 0}

    other = Ledger(ledger.db.path)
    other.observe_usage(request_id, "reported", {"input": 200}, "correction")
    other.finish_request(request_id, "completed")
    result = view.summary()
    assert result["input"] == {"verified": 200, "estimate": 0}
    assert result["cost"]["CNY"] == pytest.approx(4.1)
    assert result == view.summary()

    mirrored = ledger.get_request(request_id)
    other.mirror_request(mirrored)
    assert view.summary() == result
    request(other)
    assert view.summary()["input"]["verified"] == 300


def test_price_publication_notifies_display_and_future_plan_waits_until_effective(tmp_path):
    from TIYA.llm.usage_summary import MonthlyOverview

    ledger = Ledger(tmp_path / "usage.sqlite3")
    prices = PriceCatalog(tmp_path / "prices.sqlite3")
    request(ledger)
    notifications = []
    view = MonthlyOverview(ledger, prices, callback=lambda: notifications.append(True))
    now = datetime.now().timestamp()
    assert view.summary(at=now)["cost"] == {}
    publish(prices, effective=now + 10)
    assert notifications == [True]
    assert view.summary(at=now)["unknown_cost_requests"] == 1
    assert view.summary(at=now + 11)["cost"]["CNY"] == pytest.approx(2.1)


def test_zero_price_missing_meter_and_unrelated_identity(tmp_path):
    from TIYA.llm.usage_summary import MonthlyOverview

    ledger = Ledger(tmp_path / "usage.sqlite3")
    prices = PriceCatalog(tmp_path / "prices.sqlite3")
    request(ledger, identity={**IDENTITY, "provider": "chat_completions:other:m"})
    call = ledger.begin_call()
    pending = ledger.begin_request(call, IDENTITY, {"input": 100})
    view = MonthlyOverview(ledger, prices)
    publish(prices)
    assert view.summary()["unknown_cost_requests"] == 2

    prices.publish(
        REF, {"currency": "CNY", "tiers": [{"rates": {"input": "0"}}]}, source="free"
    )
    result = view.summary()
    assert result["cost"] == {"CNY": 0.0}
    assert result["unknown_cost_requests"] == 1
    assert ledger.get_request(pending)["charge"] is None


def test_idle_refresh_reads_no_request_rows_and_update_reads_only_changed_id(tmp_path, monkeypatch):
    from TIYA.llm.usage_summary import MonthlyOverview

    ledger = Ledger(tmp_path / "usage.sqlite3")
    first = request(ledger)
    request(ledger)
    view = MonthlyOverview(ledger)
    view.summary()
    statements = []
    import sqlite3

    connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    view.summary()
    assert not statements
    ledger.observe_usage(first, "reported", {"input": 110}, "correction")
    statements.clear()
    assert view.summary()["input"]["verified"] == 210
    reads = [sql for sql in statements if sql.startswith("SELECT") and "FROM requests" in sql]
    assert len(reads) == 1
    assert " IN (" in reads[0]
    assert first in reads[0]


def test_month_rollover_restart_and_imported_history(tmp_path):
    from TIYA.llm.usage_summary import MonthlyOverview

    ledger = Ledger(tmp_path / "usage.sqlite3")
    prices = PriceCatalog(tmp_path / "prices.sqlite3")
    request_id = request(ledger)
    with ledger.db.transaction() as db:
        db.execute("UPDATE requests SET created=? WHERE id=?", (datetime(2026, 10, 31).timestamp(), request_id))

    view = MonthlyOverview(ledger, prices)
    assert view.summary(at=datetime(2026, 10, 31, 12).timestamp())["input"]["verified"] == 100
    assert view.summary(at=datetime(2026, 11, 1).timestamp())["input"]["verified"] == 0
    ledger.import_aggregate(
        {"storage": {"2026": {"11": {"1": {"old": {
            "usage": {"input": {"verified": 50}}, "price": None,
        }}}}}},
        "old",
    )
    assert view.summary(at=datetime(2026, 11, 1).timestamp())["input"]["verified"] == 50
    restarted = MonthlyOverview(Ledger(ledger.db.path), prices)
    assert restarted.summary(at=datetime(2026, 11, 1).timestamp()) == view.summary(at=datetime(2026, 11, 1).timestamp())


def test_log_and_usage_now_refresh_on_publish_without_rewriting_accounting(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from TIYA.llm import usage_compat

    prices = PriceCatalog(tmp_path / "prices.sqlite3")
    usage = usage_compat.LLMUsage(tmp_path / "usage.sqlite3", prices=prices)
    lines = []
    monkeypatch.setattr(usage_compat, "_GLOBAL_USAGE", usage)
    monkeypatch.setattr(usage_compat, "_status", SimpleNamespace(update=lines.append))
    usage.callback = usage_compat._auto_flush
    request_id = request(usage.ledger)
    before = usage.ledger.get_request(request_id)
    expected = (
        "本月Token消耗 | 总: [110]; 输入: [100]; 输出: [10]; "
        "输入(命中缓存): [20]; 缓存命中率: [20.00%]"
    )
    assert lines[-1] == expected

    publish(PriceCatalog(prices.db.path))
    assert lines[-1] == expected + " | CNY: [2.10]"
    assert usage_compat.usage_now().CNY == pytest.approx(2.1)
    assert usage.ledger.get_request(request_id) == before
    assert usage.overview("ALL")["cost"] == {}


def test_tier_is_selected_per_request_not_by_monthly_total(tmp_path):
    from TIYA.llm.usage_summary import MonthlyOverview

    ledger = Ledger(tmp_path / "usage.sqlite3")
    prices = PriceCatalog(tmp_path / "prices.sqlite3")
    request(ledger, input_tokens=80)
    request(ledger, input_tokens=200)
    view = MonthlyOverview(ledger, prices)
    prices.publish(
        REF,
        {
            "currency": "CNY",
            "threshold": "input",
            "units": {"input_uncached": 100, "cached": 100, "output": 100},
            "tiers": [
                {"up_to": 128, "rates": {"input_uncached": "2", "cached": "0.5", "output": "4"}},
                {"rates": {"input_uncached": "4", "cached": "0.5", "output": "4"}},
            ],
        },
        source="test",
    )
    assert view.summary()["cost"]["CNY"] == pytest.approx(9.4)


def test_image_meter_and_currency_are_independent(tmp_path):
    from TIYA.llm.usage_summary import MonthlyOverview

    ledger = Ledger(tmp_path / "usage.sqlite3")
    prices = PriceCatalog(tmp_path / "prices.sqlite3")
    request(ledger)
    image_ref = "image_generation:account:image"
    image_identity = {**IDENTITY, "provider": image_ref, "model": "image", "operation": "images_generate"}
    ledger.begin_request(ledger.begin_call(), image_identity, {"images": 2})
    publish(prices)
    prices.publish(
        image_ref,
        {"currency": "USD", "threshold": "images", "tiers": [{"rates": {"images": "3"}}]},
        source="test",
    )
    result = MonthlyOverview(ledger, prices).summary()
    assert result["cost"] == {"CNY": pytest.approx(2.1), "USD": 6.0}


def test_missing_usage_with_existing_price_is_not_revalued_at_new_price(tmp_path):
    from TIYA.llm.usage_summary import MonthlyOverview

    ledger = Ledger(tmp_path / "usage.sqlite3")
    prices = PriceCatalog(tmp_path / "prices.sqlite3")
    publish(prices)
    request_id = ledger.begin_request(
        ledger.begin_call(), IDENTITY, {"input": 100}, price_snapshot=prices.snapshot(REF)
    )
    ledger.observe_usage(request_id, "reported", {"input": 100}, "partial")
    prices.publish(
        REF,
        {"currency": "CNY", "tiers": [{"rates": {"input": "0"}}]},
        source="free-now",
    )
    result = MonthlyOverview(ledger, prices).summary()
    assert result["unknown_cost_requests"] == 1
    assert result["cost"] == {}


def test_price_failure_and_bad_display_callback_do_not_corrupt_records(tmp_path, monkeypatch):
    from TIYA.llm.usage_summary import MonthlyOverview

    ledger = Ledger(tmp_path / "usage.sqlite3")
    prices = PriceCatalog(tmp_path / "prices.sqlite3")
    request(ledger)

    def broken_callback():
        raise RuntimeError("display unavailable")

    view = MonthlyOverview(ledger, prices, callback=broken_callback)
    publish(prices)
    original = prices.active_snapshots
    monkeypatch.setattr(prices, "active_snapshots", lambda **kwargs: (_ for _ in ()).throw(OSError("offline")))
    assert view.summary()["unknown_cost_requests"] == 1
    monkeypatch.setattr(prices, "active_snapshots", original)
    assert view.summary()["cost"]["CNY"] == pytest.approx(2.1)


def test_concurrent_writers_and_callbacks_preserve_totals(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from TIYA.llm.usage_summary import MonthlyOverview

    ledger = Ledger(tmp_path / "usage.sqlite3")
    view = MonthlyOverview(ledger)
    view.summary()

    def write(index):
        writer = Ledger(ledger.db.path, callback=view.summary)
        return request(writer, input_tokens=100 + index)

    with ThreadPoolExecutor(max_workers=4) as pool:
        identifiers = list(pool.map(write, range(12)))

    assert len(set(identifiers)) == 12
    assert view.summary()["input"] == {"verified": sum(range(100, 112)), "estimate": 0}
    assert view.summary()["cache_basis"] == {"input": sum(range(100, 112)), "cached": 240}


def test_real_zero_replaces_estimate_and_returned_summary_is_independent(tmp_path):
    from TIYA.llm.usage_summary import MonthlyOverview

    ledger = Ledger(tmp_path / "usage.sqlite3")
    request_id = ledger.begin_request(ledger.begin_call(), {}, {"input": 100})
    view = MonthlyOverview(ledger)
    assert view.summary()["input"] == {"verified": 0, "estimate": 100}
    ledger.observe_usage(request_id, "reported", {"input": 0}, "zero")
    result = view.summary()
    assert result["input"] == {"verified": 0, "estimate": 0}
    result["input"]["verified"] = 999
    assert view.summary()["input"]["verified"] == 0


def test_december_to_january_releases_previous_month_cache(tmp_path):
    from TIYA.llm.usage_summary import MonthlyOverview

    ledger = Ledger(tmp_path / "usage.sqlite3")
    request_id = request(ledger)
    with ledger.db.transaction() as db:
        db.execute(
            "UPDATE requests SET created=? WHERE id=?",
            (datetime(2026, 12, 31).timestamp(), request_id),
        )

    view = MonthlyOverview(ledger)
    assert view.summary(at=datetime(2026, 12, 31).timestamp())["input"]["verified"] == 100
    assert view.summary(at=datetime(2027, 1, 1).timestamp())["input"]["verified"] == 0
