import pytest

from financial_news.config import AnalysisConfig, Config, LoggingConfig, load_config


def test_defaults_when_no_file(tmp_path):
    cfg = load_config(path=tmp_path / "config.toml")
    assert cfg.logging.log_dir == "logs"
    assert cfg.logging.max_bytes == 10 * 1024 * 1024
    assert cfg.logging.filename == "financial_news"
    assert cfg.logging.level == "INFO"


def test_values_loaded_from_file(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "[logging]\n"
        'log_dir = "custom_logs"\n'
        "max_bytes = 5242880\n"
        'filename = "my_app"\n'
        'level = "DEBUG"\n'
    )
    cfg = load_config(path=config_file)
    assert cfg.logging.log_dir == "custom_logs"
    assert cfg.logging.max_bytes == 5242880
    assert cfg.logging.filename == "my_app"
    assert cfg.logging.level == "DEBUG"


def test_partial_overrides_use_defaults_for_missing_keys(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text('[logging]\nlevel = "DEBUG"\n')
    cfg = load_config(path=config_file)
    assert cfg.logging.level == "DEBUG"
    assert cfg.logging.log_dir == "logs"
    assert cfg.logging.filename == "financial_news"


def test_level_is_normalised_to_uppercase(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text('[logging]\nlevel = "debug"\n')
    cfg = load_config(path=config_file)
    assert cfg.logging.level == "DEBUG"


def test_invalid_level_raises(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text('[logging]\nlevel = "VERBOSE"\n')
    with pytest.raises(ValueError, match="level must be INFO or DEBUG"):
        load_config(path=config_file)


def test_unrecognised_key_raises(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text('[logging]\nunknown_key = "value"\n')
    with pytest.raises(ValueError, match="unrecognised keys"):
        load_config(path=config_file)


def test_empty_logging_section_uses_defaults(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[logging]\n")
    cfg = load_config(path=config_file)
    assert cfg == Config(logging=LoggingConfig())


def test_missing_logging_section_uses_defaults(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("# no sections\n")
    cfg = load_config(path=config_file)
    assert cfg == Config(logging=LoggingConfig())


# ---------------------------------------------------------------------------
# [analysis] section
# ---------------------------------------------------------------------------


def test_analysis_defaults_when_no_file(tmp_path):
    cfg = load_config(path=tmp_path / "config.toml")
    assert cfg.analysis.baseline_days == 30
    assert cfg.analysis.threshold_elevated == 2.0
    assert cfg.analysis.threshold_unusual == 3.0


def test_analysis_values_loaded_from_file(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "[analysis]\n"
        "baseline_days = 20\n"
        "threshold_elevated = 1.5\n"
        "threshold_unusual = 2.5\n"
    )
    cfg = load_config(path=config_file)
    assert cfg.analysis.baseline_days == 20
    assert cfg.analysis.threshold_elevated == 1.5
    assert cfg.analysis.threshold_unusual == 2.5


def test_analysis_partial_overrides_use_defaults(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[analysis]\nbaseline_days = 60\n")
    cfg = load_config(path=config_file)
    assert cfg.analysis.baseline_days == 60
    assert cfg.analysis.threshold_elevated == AnalysisConfig.threshold_elevated
    assert cfg.analysis.threshold_unusual == AnalysisConfig.threshold_unusual


def test_analysis_unrecognised_key_raises(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[analysis]\nunknown_key = 99\n")
    with pytest.raises(ValueError, match="unrecognised keys"):
        load_config(path=config_file)


def test_analysis_baseline_days_zero_raises(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[analysis]\nbaseline_days = 0\n")
    with pytest.raises(ValueError, match="baseline_days must be > 0"):
        load_config(path=config_file)


def test_analysis_threshold_ordering_raises(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "[analysis]\nthreshold_elevated = 3.0\nthreshold_unusual = 2.0\n"
    )
    with pytest.raises(ValueError, match="threshold_elevated.*must be less than"):
        load_config(path=config_file)
