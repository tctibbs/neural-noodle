"""Lockstep equivalence tests pinning the tensor env to the numpy
reference implementation.
"""

import numpy as np
import pytest
import torch

from snake_rl.config import BoardSpec, EnvConfig, ObsConfig
from snake_rl.env.tensor_env import TensorVecSnake
from snake_rl.env.vec_env import DOWN, LEFT, RIGHT, UP, VecSnake
from snake_rl.obs import GridObsBuilder


def make_pair(
    width: int = 8,
    height: int = 8,
    num_envs: int = 16,
    seed: int = 3,
    start_length: int = 3,
) -> tuple[VecSnake, TensorVecSnake]:
    """Build a numpy env and a tensor env loaded with its state."""
    config = EnvConfig(
        boards=[BoardSpec(width=width, height=height)],
        num_envs=num_envs,
        start_length=start_length,
        starvation_factor=4.0,
        init_mode="random",
    )
    ref = VecSnake(config, seed=seed)
    tens = TensorVecSnake(
        config,
        ObsConfig(),
        canvas=max(width, height),
        seed=seed,
        device="cpu",
    )
    tens.load_state_from(ref)
    return ref, tens


def assert_same_state(
    ref: VecSnake, tens: TensorVecSnake, skip: np.ndarray
) -> None:
    """Compare core state for all envs not in the skip mask."""
    keep = ~skip
    occ_t = tens.occ.numpy().reshape(ref.occ.shape)
    assert (occ_t[keep] == ref.occ[keep]).all()
    assert (tens.length.numpy()[keep] == ref.length[keep]).all()
    assert (tens.direction.numpy()[keep] == ref.direction[keep]).all()
    assert (tens.head.numpy()[keep] == ref.head[keep]).all()
    assert (tens.steps.numpy()[keep] == ref.steps[keep]).all()
    assert (tens.apples.numpy()[keep] == ref.apples[keep]).all()


@pytest.mark.parametrize(("width", "height"), [(8, 8), (6, 10), (12, 6)])
def test_lockstep_equivalence(width: int, height: int) -> None:
    """Single-step transitions match the reference from any state.

    Both envs step with identical actions from identical pre-states.
    Event flags must match exactly; post-state must match for every
    env that did not reset (resets draw from different RNG streams,
    so the tensor env is resynced from the reference each step).
    """
    ref, tens = make_pair(width=width, height=height)
    rng = np.random.default_rng(0)
    for _ in range(400):
        actions = rng.integers(0, 4, ref.num_envs)
        result = ref.step(actions)
        events = tens.step(torch.from_numpy(actions))
        assert (events["ate"].numpy() == result.ate).all()
        assert (events["died"].numpy() == result.died).all()
        assert (events["won"].numpy() == result.won).all()
        assert (events["starved"].numpy() == result.starved).all()
        assert_same_state(ref, tens, skip=result.done)
        tens.load_state_from(ref)
    assert ref.drain_finished(), "no episodes finished; test too weak"


def test_win_on_full_board() -> None:
    """Filling a 2x2 board wins and resets in the tensor env."""
    config = EnvConfig(
        boards=[BoardSpec(width=2, height=2)],
        num_envs=1,
        start_length=1,
        starvation_factor=100.0,
        init_mode="row0",
    )
    tens = TensorVecSnake(config, ObsConfig(), canvas=2, seed=5, device="cpu")
    ring = {(0, 0): RIGHT, (0, 1): DOWN, (1, 1): LEFT, (1, 0): UP}
    won = False
    for _ in range(60):
        r, c = (int(x) for x in tens.head[0])
        events = tens.step(torch.tensor([ring[(r, c)]]))
        if bool(events["won"][0]):
            won = True
            break
    assert won
    stats = tens.drain_finished()
    assert stats[-1].won
    assert stats[-1].length == 4


def test_determinism() -> None:
    """Same seed and actions give identical trajectories."""
    config = EnvConfig(
        boards=[BoardSpec(width=8, height=8)],
        num_envs=8,
        start_length=3,
        init_mode="random",
    )
    a = TensorVecSnake(config, ObsConfig(), canvas=8, seed=9, device="cpu")
    b = TensorVecSnake(config, ObsConfig(), canvas=8, seed=9, device="cpu")
    rng = np.random.default_rng(4)
    for _ in range(200):
        actions = torch.from_numpy(rng.integers(0, 4, 8))
        a.step(actions.clone())
        b.step(actions.clone())
    assert torch.equal(a.body, b.body)
    assert torch.equal(a.fruit, b.fruit)
    assert torch.equal(a.length, b.length)


@pytest.mark.parametrize("egocentric", [False, True])
@pytest.mark.parametrize("body_decay", [False, True])
def test_observe_matches_numpy_builder(
    egocentric: bool, body_decay: bool
) -> None:
    """Fused observe() reproduces the numpy grid observation."""
    obs_config = ObsConfig(egocentric=egocentric, body_decay=body_decay)
    config = EnvConfig(
        boards=[BoardSpec(width=8, height=8)],
        num_envs=16,
        start_length=3,
        init_mode="random",
    )
    ref = VecSnake(config, seed=11)
    tens = TensorVecSnake(config, obs_config, canvas=8, seed=11, device="cpu")
    builder = GridObsBuilder(obs_config, canvas=8)
    rng = np.random.default_rng(6)
    for _ in range(50):
        actions = rng.integers(0, 4, 16)
        ref.step(actions)
        tens.load_state_from(ref)
        expected = builder.build(ref)
        got = tens.observe().numpy()
        np.testing.assert_allclose(got, expected, atol=1e-6)
