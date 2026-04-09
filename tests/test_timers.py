from app.utils.timers import CooldownTimer, LoopRegulator, RateLimiter


def test_cooldown() -> None:
    timer = CooldownTimer(1.0)
    assert timer.ready(2.0) is True
    timer.mark(2.0)
    assert timer.ready(2.5) is False
    assert timer.ready(3.2) is True


def test_rate_limiter() -> None:
    limiter = RateLimiter(2.0)
    assert limiter.allow(1.0) is True
    assert limiter.allow(1.1) is False
    assert limiter.allow(1.6) is True


def test_loop_regulator_remaining_sleep() -> None:
    regulator = LoopRegulator(5.0)
    assert regulator.target_period_seconds == 0.2
    assert regulator.remaining_sleep(0.05) == 0.15


def test_loop_regulator_no_double_sleep_on_overrun() -> None:
    regulator = LoopRegulator(5.0)
    assert regulator.remaining_sleep(0.25) == 0.0
