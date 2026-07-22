from TIYA.model.attention_speak import AttentionSpeakProbability


def test_probability_bounds_always_valid():
    model = AttentionSpeakProbability()
    p_min, p_max, win = 0.08, 0.76, 300

    for t in range(0, 1200):
        if (t % 53 == 0) or (t % 71 == 0):
            model.wake(now_ts=float(t), active_window_s=win)
        p = model.get_probability(prob_min=p_min, prob_max=p_max, active_window_s=win, now_ts=float(t))
        assert p_min <= p <= p_max


def test_sleep_timeout_then_wake_jump():
    model = AttentionSpeakProbability()
    p_min, p_max, win = 0.10, 0.80, 300

    # Wake into active state
    model.wake(now_ts=0.0, active_window_s=win)
    p1 = model.get_probability(p_min, p_max, win, now_ts=0.0)
    assert p1 > p_min

    # Timeout beyond dynamic window -> should sleep back near min
    p2 = model.get_probability(p_min, p_max, win, now_ts=2000.0)
    assert abs(p2 - p_min) < 0.03

    # Wake again -> obvious jump
    model.wake(now_ts=2035.0, active_window_s=win)
    p3 = model.get_probability(p_min, p_max, win, now_ts=2035.0)
    assert p3 - p2 > 0.08


def test_penalty_drops_probability_under_frequent_wakes():
    model = AttentionSpeakProbability()
    p_min, p_max, win = 0.05, 0.85, 300

    samples = []
    t = 0.0
    for _ in range(14):
        t += 8.0
        model.wake(now_ts=t, active_window_s=win)
        samples.append(model.get_probability(p_min, p_max, win, now_ts=t))

    assert max(samples) > p_min + 0.12
    assert samples[-1] < max(samples) - 0.02


def test_time_normalization_reduces_frequency_coupling_when_min_is_zero():
    p_min, p_max, win = 0.0, 0.8, 300
    duration_s = 600.0

    def expected_rate(call_dt: float) -> float:
        model = AttentionSpeakProbability(enable_time_normalization=True)
        model.attention = 0.82
        model.fatigue = 0.27
        model.active = False

        t = 0.0
        expected_count = 0.0
        while t <= duration_s:
            p = model.get_probability(p_min, p_max, win, now_ts=t)
            expected_count += p
            t += call_dt
        return expected_count / duration_s

    rate_fast = expected_rate(0.2)
    rate_slow = expected_rate(2.0)
    rel_gap = abs(rate_fast - rate_slow) / max(rate_fast, rate_slow, 1e-8)
    assert rel_gap < 0.25


def test_new_api_wake_and_get_probability_works_without_explicit_advance():
    model = AttentionSpeakProbability(min_prob=0.05, max_prob=0.75, time_window=300)
    p0 = model.get_probability(now_ts=0.0)
    assert p0 >= 0.05

    model.wake(now_ts=60.0)
    p1 = model.get_probability(now_ts=60.0)
    assert p1 > p0

    # No explicit advance call here; get_probability should still handle time progression internally.
    p2 = model.get_probability(now_ts=2000.0)
    assert abs(p2 - 0.05) < 0.03
