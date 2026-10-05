import math

import pytest

from simulations.environment import Environment
from simulations.room_model import step_room
from utils.rng import RNG


def test_environment_oscillates_and_applies_door_drop():
    env = Environment(
        base=10.0,
        amplitude=2.0,
        period_s=100.0,
        door_drop_C=5.0,
        door_start_s=20.0,
        door_duration_s=10.0,
    )

    assert env.T_out(0.0) == pytest.approx(10.0)
    assert env.T_out(25.0) == pytest.approx(12.0)
    assert env.T_out(20.0) == pytest.approx(10.0 + 2.0 * math.sin(0.4 * math.pi) - 5.0)
    assert env.T_out(30.0) == pytest.approx(10.0 + 2.0 * math.sin(0.6 * math.pi))


def test_environment_rejects_non_positive_period():
    with pytest.raises(ValueError, match="period_s"):
        Environment(period_s=0.0).T_out(0.0)


def test_room_model_uses_euler_step():
    result = step_room(
        T=20.0,
        heater_on=1,
        T_out=10.0,
        R=2.0,
        C=10.0,
        P=20.0,
        dt=1.0,
    )

    assert result == pytest.approx(21.5)


def test_room_model_requires_rng_for_process_noise():
    with pytest.raises(ValueError, match="rng is required"):
        step_room(20.0, 0, 10.0, 2.0, 10.0, 20.0, 1.0, process_sigma=0.1)


def test_random_generator_is_seeded_and_bernoulli_is_bounded():
    first = RNG(seed=7)
    second = RNG(seed=7)

    assert first.gauss(0.0, 1.0) == second.gauss(0.0, 1.0)
    assert first.uniform(-2.0, 3.0) == second.uniform(-2.0, 3.0)
    assert RNG(seed=1).bernoulli(0.0) is False
    assert RNG(seed=1).bernoulli(1.0) is True

    with pytest.raises(ValueError, match="between 0 and 1"):
        RNG(seed=1).bernoulli(1.1)
