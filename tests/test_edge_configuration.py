"""Tests for GEONAV-AI edge configuration / operating profiles (Phase 8 B14)."""

import pytest
from geonav.config.settings import (
    OperatingProfile,
    ProfileConfig,
    Settings,
    PROFILE_CONFIGS,
    get_settings_for_profile,
)


# ── OperatingProfile enum ────────────────────────────────────────────────────

def test_all_profiles_exist():
    """Test: All three operating profiles are defined."""
    profiles = list(OperatingProfile)
    assert OperatingProfile.DEVELOPMENT in profiles
    assert OperatingProfile.BALANCED in profiles
    assert OperatingProfile.EDGE in profiles


def test_profile_values_are_strings():
    """Test: Operating profile enum values are non-empty strings."""
    for p in OperatingProfile:
        assert isinstance(p.value, str)
        assert len(p.value) > 0


# ── ProfileConfig dataclass ───────────────────────────────────────────────────

def test_profile_config_defaults():
    """Test: Default ProfileConfig matches DEVELOPMENT settings."""
    cfg = ProfileConfig()
    assert cfg.max_features == 200
    assert cfg.image_scale == pytest.approx(1.0)
    assert cfg.mapping_enabled is True
    assert cfg.ml_inference_every_n == 1
    assert cfg.keyframe_interval_frames == 15
    assert cfg.trajectory_buffer_limit == 0


def test_profile_config_custom_values():
    """Test: ProfileConfig accepts custom parameter overrides."""
    cfg = ProfileConfig(
        max_features=100,
        image_scale=1.0,
        mapping_enabled=False,
        ml_inference_every_n=3,
        keyframe_interval_frames=30,
        trajectory_buffer_limit=5000,
    )
    assert cfg.max_features == 100
    assert cfg.mapping_enabled is False
    assert cfg.ml_inference_every_n == 3


# ── PROFILE_CONFIGS dictionary ────────────────────────────────────────────────

def test_all_profiles_have_config():
    """Test: Every OperatingProfile has a corresponding PROFILE_CONFIGS entry."""
    for p in OperatingProfile:
        assert p in PROFILE_CONFIGS, f"No ProfileConfig for {p}"


def test_development_profile_max_features():
    """Test: DEVELOPMENT profile uses maximum feature count."""
    cfg_dev = PROFILE_CONFIGS[OperatingProfile.DEVELOPMENT]
    cfg_edge = PROFILE_CONFIGS[OperatingProfile.EDGE]
    assert cfg_dev.max_features >= cfg_edge.max_features


def test_edge_profile_disables_mapping():
    """Test: EDGE profile disables mapping subsystem."""
    cfg = PROFILE_CONFIGS[OperatingProfile.EDGE]
    assert cfg.mapping_enabled is False


def test_edge_profile_lower_ml_frequency():
    """Test: EDGE profile runs ML inference less frequently than DEVELOPMENT."""
    cfg_dev = PROFILE_CONFIGS[OperatingProfile.DEVELOPMENT]
    cfg_edge = PROFILE_CONFIGS[OperatingProfile.EDGE]
    assert cfg_edge.ml_inference_every_n >= cfg_dev.ml_inference_every_n


def test_edge_profile_bounded_trajectory_buffer():
    """Test: EDGE profile has a finite trajectory buffer limit."""
    cfg = PROFILE_CONFIGS[OperatingProfile.EDGE]
    assert cfg.trajectory_buffer_limit > 0


def test_development_profile_unlimited_trajectory():
    """Test: DEVELOPMENT profile allows unlimited trajectory storage."""
    cfg = PROFILE_CONFIGS[OperatingProfile.DEVELOPMENT]
    assert cfg.trajectory_buffer_limit == 0  # 0 = unlimited


# ── get_settings_for_profile ─────────────────────────────────────────────────

def test_get_settings_development_profile():
    """Test: DEVELOPMENT profile settings have full VIO features and mapping enabled."""
    settings = get_settings_for_profile(OperatingProfile.DEVELOPMENT)
    assert settings.profile == OperatingProfile.DEVELOPMENT
    assert settings.vio.max_features == 200
    assert settings.mapping.enabled is True


def test_get_settings_edge_profile():
    """Test: EDGE profile settings have reduced features and mapping disabled."""
    settings = get_settings_for_profile(OperatingProfile.EDGE)
    assert settings.profile == OperatingProfile.EDGE
    assert settings.vio.max_features == 100
    assert settings.mapping.enabled is False


def test_get_settings_balanced_profile():
    """Test: BALANCED profile is between DEVELOPMENT and EDGE in features."""
    dev = get_settings_for_profile(OperatingProfile.DEVELOPMENT)
    bal = get_settings_for_profile(OperatingProfile.BALANCED)
    edg = get_settings_for_profile(OperatingProfile.EDGE)
    assert edg.vio.max_features <= bal.vio.max_features <= dev.vio.max_features


def test_get_settings_profile_config_matches():
    """Test: get_settings_for_profile propagates ProfileConfig to Settings."""
    for profile in OperatingProfile:
        settings = get_settings_for_profile(profile)
        expected_cfg = PROFILE_CONFIGS[profile]
        assert settings.profile_config.max_features == expected_cfg.max_features
        assert settings.profile_config.mapping_enabled == expected_cfg.mapping_enabled


def test_settings_default_is_development():
    """Test: Default Settings uses DEVELOPMENT profile."""
    s = Settings()
    assert s.profile == OperatingProfile.DEVELOPMENT


def test_profile_reproducibility():
    """Test: Calling get_settings_for_profile twice produces identical settings."""
    s1 = get_settings_for_profile(OperatingProfile.EDGE)
    s2 = get_settings_for_profile(OperatingProfile.EDGE)
    assert s1.vio.max_features == s2.vio.max_features
    assert s1.mapping.enabled == s2.mapping.enabled
    assert s1.profile_config.ml_inference_every_n == s2.profile_config.ml_inference_every_n
