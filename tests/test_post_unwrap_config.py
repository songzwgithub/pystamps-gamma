from __future__ import annotations

import pytest

from pystamps.config import (
    ConfigError,
    PostUnwrapDerampConfig,
)


def test_post_unwrap_deramp_defaults_off():
    cfg = PostUnwrapDerampConfig()
    assert cfg.enabled is False
    assert cfg.mode == "linear_spacetime"


def test_post_unwrap_deramp_rejects_unknown_mode():
    with pytest.raises(ConfigError):
        PostUnwrapDerampConfig(
            mode="per_epoch_full_plane"
        )
