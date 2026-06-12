"""Tests for the configuration schema."""

from pathlib import Path

from snake_rl.config import (
    BoardSpec,
    TrainConfig,
    config_hash,
    load_config,
    save_config,
)


def test_hash_is_stable_and_sensitive() -> None:
    """Equal configs hash equal; any field change rehashes."""
    a = TrainConfig()
    b = TrainConfig()
    assert config_hash(a) == config_hash(b)
    c = TrainConfig(reward=a.reward.model_copy(update={"step_cost": 0.02}))
    assert config_hash(a) != config_hash(c)


def test_yaml_round_trip(tmp_path: Path) -> None:
    """Configs survive YAML serialization unchanged."""
    config = TrainConfig(
        run_name="round-trip",
        seed=42,
        env=TrainConfig().env.model_copy(
            update={"boards": [BoardSpec(width=12, height=10)]}
        ),
    )
    path = tmp_path / "config.yaml"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded == config
    assert config_hash(loaded) == config_hash(config)
