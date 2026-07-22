from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class AttentionSpeakProbability:
    """Stateful probability generator that simulates group-chat attention dynamics.

    Core behavior:
    - Wake from sleep quickly boosts attention.
    - A few consecutive wakes can reach near-max activity.
    - Frequent wakes and active-time fatigue pull probability down gradually.
    - If no wake arrives in window, it drops to sleep.
    - Wakes can extend window, but session length is hard-capped.
    """

    # 外部配置参数
    min_prob: float = 0.00
    max_prob: float = 1.00
    time_window: int = 300

    # 唤醒增长参数（安全区间）
    # - sleep_jump: [0.35, 0.65] 沉睡被唤醒后的第一跳高度
    # - wake_gain: [0.45, 0.85] 连续 wake 的主增益（决定“几次聊进去”）
    # - cadence_gain: [0.10, 0.35] 短间隔 wake 额外加成（更像“对上节奏”）
    # - cadence_tau_s: [20, 70] 节奏记忆时间常数
    sleep_jump: float = 0.40
    wake_gain: float = 0.60
    cadence_gain: float = 0.27
    cadence_tau_s: float = 40.00

    # 疲劳抑制参数（收敛核心）
    # - fatigue_gain: [0.05, 0.25] 每次 wake 的疲劳增量
    # - fatigue_time_ratio: [1.00, 3.00] 活跃中纯时间疲劳，越小越快聊累
    # - fatigue_penalty_strength: [0.70, 1.00] 疲劳压制强度
    # - short_gap_tau_s: [15, 40] 高频 wake 惩罚敏感度
    # - fatigue_recover_tau_s: [180, 1200] 沉睡恢复速度
    fatigue_gain: float = 0.08
    fatigue_time_ratio: float = 2.7
    fatigue_penalty_strength: float = 0.86
    short_gap_tau_s: float = 32.00
    fatigue_recover_tau_s: float = 510.00

    # 连续唤醒记忆
    # - streak_decay_tau_s: [60, 260] 越大越容易保持“连聊记忆”
    # - streak_saturation: [2.50, 6.00] 越小越快达到“已连聊”状态
    streak_decay_tau_s: float = 210
    streak_saturation: float = 4.20

    # 活跃窗口扩展约束
    # - max_window_extension_ratio: [0.50, 2.00]
    #   单次 wake 后的最大 gap 窗口 = base_window * (1 + ratio)
    # - max_session_extension_ratio: [1.00, 3.00]
    #   整个活跃会话最大总时长 = base_window * (1 + ratio)，用于硬防无限续命
    max_window_extension_ratio: float = 0.50
    max_session_extension_ratio: float = 1.50

    # 活跃态自然衰减
    # - attention_decay_ratio: [1.00, 2.20] 越小衰减越快
    attention_decay_ratio: float = 1.40

    # 时间归一化（跨调用频率一致性）
    # - reference_window_ratio: [0.08, 0.30]
    # - min_reference_s: [1.50, 8.00]
    # - max_reference_ratio: [0.60, 2.40]
    # - reference_adapt_gain: [0.45, 1.35]
    reference_window_ratio: float = 0.14
    min_reference_s: float = 4.70
    max_reference_ratio: float = 1.10
    reference_adapt_gain: float = 1.10
    # 是否启用按 dt 的时间归一化概率。你的“队列串行+20~60s决策”场景建议关闭（默认）。
    enable_time_normalization: bool = False

    # 连续 wake 负载惩罚（活跃期只增不减，防反弹）
    # - wake_load_gain: [0.01, 0.20] 累积唤醒次数对概率的抑制
    # - wake_load_power: [1.00, 3.50] 抑制曲线形状，越大后段压得越明显
    # - sleep_reset_factor: [0.10, 0.60] 进入沉睡时保留疲劳比例
    wake_load_gain: float = 0.05
    wake_load_power: float = 2.50
    sleep_reset_factor: float = 0.20

    # 冷启动/预热参数（保证“沉睡 -> 峰值”需要少量 wake 过渡）
    # - warmup_k: [1.20, 3.50] 越大越慢进入峰值，首条 wake 越不容易贴近最大
    # - sleep_jump_to_peak_ratio: [0.35, 0.80] 首跳相对“当前可达峰值”的比例
    warmup_k: float = 2.2
    sleep_jump_to_peak_ratio: float = 0.45

    # Internal state
    attention: float = 0.0
    fatigue: float = 0.0
    wake_streak: float = 0.0
    wake_load: float = 0.0
    active: bool = False

    _last_tick_ts: float | None = field(default=None, init=False)
    _last_wake_ts: float | None = field(default=None, init=False)
    _dynamic_window_s: float = field(default=0.0, init=False)
    _last_reference_s: float = field(default=0.0, init=False)
    _session_start_ts: float | None = field(default=None, init=False)
    _session_deadline_ts: float | None = field(default=None, init=False)

    def _wake_load_penalty(self) -> float:
        return 1.0 / (1.0 + self.wake_load_gain * (max(0.0, self.wake_load) ** self.wake_load_power))

    def _warmup_gate(self) -> float:
        return 1.0 - math.exp(-max(0.0, self.wake_streak) / max(1e-6, self.warmup_k))

    @staticmethod
    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def reset(self) -> None:
        """Reset all internal attention states."""
        self.attention = 0.0
        self.fatigue = 0.0
        self.wake_streak = 0.0
        self.wake_load = 0.0
        self.active = False
        self._last_tick_ts = None
        self._last_wake_ts = None
        self._dynamic_window_s = 0.0
        self._last_reference_s = 0.0
        self._session_start_ts = None
        self._session_deadline_ts = None

    def _update_window(self, active_window_s: int, cadence_drive: float) -> None:
        streak_drive = 1.0 - math.exp(-self.wake_streak / self.streak_saturation)
        extension_drive = 0.60 * streak_drive + 0.40 * cadence_drive
        extension_ratio = self.max_window_extension_ratio * extension_drive
        self._dynamic_window_s = float(active_window_s) * (1.0 + extension_ratio)

    def _extend_session_deadline(self, now_ts: float, active_window_s: int) -> None:
        if self._session_start_ts is None:
            return
        hard_cap_s = float(active_window_s) * (1.0 + self.max_session_extension_ratio)
        hard_deadline = self._session_start_ts + hard_cap_s
        candidate = now_ts + max(float(active_window_s), self._dynamic_window_s)
        if self._session_deadline_ts is None:
            self._session_deadline_ts = min(hard_deadline, candidate)
        else:
            self._session_deadline_ts = min(hard_deadline, max(self._session_deadline_ts, candidate))

    def _enter_sleep(self) -> None:
        self.attention = 0.0
        self.wake_streak = 0.0
        self.wake_load = 0.0
        self.active = False
        self._last_wake_ts = None
        self._session_start_ts = None
        self._session_deadline_ts = None
        self.fatigue *= self.sleep_reset_factor

    def _check_timeout(self, now_ts: float, active_window_s: int) -> None:
        if not self.active:
            return

        if self._session_deadline_ts is not None and now_ts >= self._session_deadline_ts:
            self._enter_sleep()
            return

        if self._last_wake_ts is None:
            self._enter_sleep()
            return

        timeout_window = self._dynamic_window_s if self._dynamic_window_s > 0 else float(active_window_s)
        if now_ts - self._last_wake_ts > timeout_window:
            self._enter_sleep()

    def _dynamic_reference_s(self, active_window_s: int) -> float:
        base_ref = max(self.min_reference_s, self.reference_window_ratio * float(active_window_s))
        max_ref = max(self.min_reference_s, self.max_reference_ratio * float(active_window_s))

        load_drive = 1.0 - math.exp(-self.wake_load / 6.0)
        activity_drive = 0.45 * self.attention + 0.35 * self.fatigue + 0.20 * load_drive
        state_scale = 1.0 + self.reference_adapt_gain * activity_drive

        ref_s = base_ref * state_scale
        return max(self.min_reference_s, min(max_ref, ref_s))

    @staticmethod
    def _hazard_normalize(base_prob: float, dt: float, ref_s: float) -> float:
        if dt <= 0.0:
            return 0.0
        if base_prob <= 0.0:
            return 0.0
        if base_prob >= 1.0:
            return 1.0

        ratio = dt / max(ref_s, 1e-6)
        # Stable form of: 1 - (1 - base_prob) ** ratio
        return 1.0 - math.exp(math.log1p(-base_prob) * ratio)

    def _resolve_window(self, active_window_s: int | None) -> int:
        if active_window_s is None:
            active_window_s = self.time_window
        if active_window_s <= 0:
            raise ValueError("active_window_s must be > 0")
        return int(active_window_s)

    @staticmethod
    def _resolve_probability_bounds(prob_min: float | None, prob_max: float | None) -> tuple[float, float]:
        if prob_min is None:
            prob_min = 0.0
        if prob_max is None:
            prob_max = 1.0
        if not (0.0 <= prob_min <= 1.0 and 0.0 <= prob_max <= 1.0):
            raise ValueError("prob_min and prob_max must be in [0, 1]")
        if prob_min > prob_max:
            raise ValueError("prob_min cannot be greater than prob_max")
        return float(prob_min), float(prob_max)

    def _resolve_now(self, now_ts: float | None) -> float:
        return time.monotonic() if now_ts is None else float(now_ts)

    def _ensure_initialized(self, now: float, active_window_s: int) -> None:
        if self._last_tick_ts is None:
            self._last_tick_ts = now
            self._dynamic_window_s = float(active_window_s)

    def advance(self, now_ts: float | None = None, active_window_s: int | None = None) -> float:
        """推进内部时间状态，不做唤醒和概率抽样。"""
        window_s = self._resolve_window(active_window_s)
        now = self._resolve_now(now_ts)
        self._ensure_initialized(now, window_s)

        dt = max(0.0, now - self._last_tick_ts)
        self._last_tick_ts = now

        # 时间推进：活跃期只累积疲劳，沉睡期恢复疲劳。
        if dt > 0.0:
            self.wake_streak *= math.exp(-dt / self.streak_decay_tau_s)
            if self.active:
                self.attention *= math.exp(-dt / (self.attention_decay_ratio * window_s))
                self.fatigue = self._clamp01(
                    self.fatigue + dt / max(1.0, self.fatigue_time_ratio * float(window_s))
                )
            else:
                self.fatigue *= math.exp(-dt / self.fatigue_recover_tau_s)

        # Rule 6: no wake in active window -> immediate sleep.
        self._check_timeout(now, window_s)
        return dt

    def wake(
        self,
        now_ts: float | None = None,
        active_window_s: int | None = None,
        wake_strength: float = 1.0,
    ) -> None:
        """应用一次唤醒事件，并更新注意力/疲劳/动态窗口。"""
        window_s = self._resolve_window(active_window_s)
        now = self._resolve_now(now_ts)
        self.advance(now_ts=now, active_window_s=window_s)

        strength = max(0.0, float(wake_strength))
        if strength <= 0.0:
            return

        if not self.active:
            self.active = True
            self._session_start_ts = now
            self._session_deadline_ts = now + float(window_s)
            self._dynamic_window_s = float(window_s)

        dt_from_last_wake = 1e9 if self._last_wake_ts is None else max(0.0, now - self._last_wake_ts)
        cadence_drive = math.exp(-dt_from_last_wake / self.cadence_tau_s)
        short_gap_drive = math.exp(-dt_from_last_wake / self.short_gap_tau_s)

        # Rule 5: strong wake-up jump from sleep.
        # 首跳按“当前可达峰值”的比例起跳，避免直接贴近实际峰值。
        fatigue_headroom = self._clamp01(1.0 - self.fatigue_penalty_strength * self.fatigue)
        curr_peak_drive = self._clamp01(fatigue_headroom * self._wake_load_penalty())
        target_jump = max(self.sleep_jump, curr_peak_drive * self.sleep_jump_to_peak_ratio)
        if self.attention < self.sleep_jump:
            self.attention = max(self.attention, target_jump)

        next_streak = self.wake_streak + strength
        wake_gain_gate = 1.0 - math.exp(-max(0.0, next_streak) / max(1e-6, self.warmup_k))
        growth = self.wake_gain * strength * (1.0 - self.attention) * wake_gain_gate
        cadence_boost = self.cadence_gain * strength * cadence_drive * (1.0 - self.attention) * wake_gain_gate
        self.attention = self._clamp01(self.attention + growth + cadence_boost)

        self.wake_streak += strength
        self.wake_load += strength * (0.55 + 0.45 * short_gap_drive)
        self._last_wake_ts = now

        # Rule 7: frequent wakes accumulate fatigue penalty.
        streak_drive = 1.0 - math.exp(-self.wake_streak / self.streak_saturation)
        fatigue_step = self.fatigue_gain * strength * streak_drive * (0.40 + 0.60 * short_gap_drive)
        self.fatigue = self._clamp01(self.fatigue + fatigue_step)

        # Rule 8: wake can extend active window, but extension is capped.
        self._update_window(window_s, cadence_drive)
        self._extend_session_deadline(now, window_s)

    def get_probability(
        self,
        prob_min: float | None = None,
        prob_max: float | None = None,
        active_window_s: int | None = None,
        now_ts: float | None = None,
        decision_dt_s: float | None = None,
    ) -> float:
        """获取当前发言概率。该接口会自动推进时间状态。"""
        window_s = self._resolve_window(active_window_s)
        lo, hi = self._resolve_probability_bounds(
            self.min_prob if prob_min is None else prob_min,
            self.max_prob if prob_max is None else prob_max,
        )
        now = self._resolve_now(now_ts)
        dt = self.advance(now_ts=now, active_window_s=window_s)

        # 显式三段式：唤醒上升 -> 少量 wake 触顶 -> fatigue/wake_load 拉回。
        effective_attention = self._clamp01(self.attention * (1.0 - self.fatigue_penalty_strength * self.fatigue))
        wake_load_penalty = self._wake_load_penalty()
        warmup_gate = self._warmup_gate()
        drive = self._clamp01(effective_attention * wake_load_penalty * warmup_gate)
        base_probability = lo + (hi - lo) * drive

        # Default behavior: return state probability directly.
        if not self.enable_time_normalization:
            return max(lo, min(hi, base_probability))

        ref_s = self._dynamic_reference_s(window_s)
        self._last_reference_s = ref_s

        span = hi - lo
        if span <= 0.0:
            return self._clamp01(lo)

        # Normalize into [0, 1], apply time-based hazard transform, then map back.
        normalized_base = self._clamp01((base_probability - lo) / span)
        if decision_dt_s is not None:
            dt_for_prob = max(0.0, float(decision_dt_s))
        else:
            dt_for_prob = dt if dt > 0.0 else ref_s
        normalized_eff = self._hazard_normalize(normalized_base, dt_for_prob, ref_s)
        probability = lo + span * normalized_eff
        return max(lo, min(hi, probability))


__all__ = ["AttentionSpeakProbability"]
