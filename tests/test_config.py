import pytest

from financial_news.config import (
    _DEFAULT_MONITOR_TICKERS,  # noqa: PLC2701
    AnalysisConfig,
    Config,
    LoggingConfig,
    MonitorConfig,
    load_config,
)


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
    assert cfg.analysis.z_score_cap == 99.0


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


def test_analysis_z_score_cap_loaded_from_file(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[analysis]\nz_score_cap = 50.0\n")
    cfg = load_config(path=config_file)
    assert cfg.analysis.z_score_cap == 50.0


def test_analysis_z_score_cap_zero_raises(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[analysis]\nz_score_cap = 0.0\n")
    with pytest.raises(ValueError, match="z_score_cap must be > 0"):
        load_config(path=config_file)


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


# ---------------------------------------------------------------------------
# [monitor] section
# ---------------------------------------------------------------------------


def test_monitor_defaults_when_no_file(tmp_path):
    cfg = load_config(path=tmp_path / "config.toml")
    assert cfg.monitor.tickers == _DEFAULT_MONITOR_TICKERS


def test_monitor_tickers_loaded_from_file(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text('[monitor]\ntickers = ["AAPL", "GOOG"]\n')
    cfg = load_config(path=config_file)
    assert cfg.monitor.tickers == ["AAPL", "GOOG"]


def test_monitor_empty_tickers_raises(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[monitor]\ntickers = []\n")
    with pytest.raises(ValueError, match="tickers must not be empty"):
        load_config(path=config_file)


def test_monitor_unrecognised_key_raises(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text('[monitor]\nunknown_key = "value"\n')
    with pytest.raises(ValueError, match="unrecognised keys"):
        load_config(path=config_file)


def test_monitor_defaults_preserved_when_only_other_sections_present(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text('[logging]\nlevel = "DEBUG"\n')
    cfg = load_config(path=config_file)
    assert cfg.monitor.tickers == _DEFAULT_MONITOR_TICKERS
    assert isinstance(cfg.monitor, MonitorConfig)


# ---------------------------------------------------------------------------
# Cross-section and edge cases
# ---------------------------------------------------------------------------


def test_all_three_sections_together(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[logging]\nlevel = "DEBUG"\n'
        "[analysis]\nbaseline_days = 60\n"
        '[monitor]\ntickers = ["AAPL"]\n'
    )
    cfg = load_config(path=config_file)
    assert cfg.logging.level == "DEBUG"
    assert cfg.analysis.baseline_days == 60
    assert cfg.monitor.tickers == ["AAPL"]


def test_unknown_top_level_section_is_silently_ignored(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[unknown_section]\nfoo = 1\n")
    cfg = load_config(path=config_file)
    assert cfg == Config()


def test_invalid_toml_raises(tmp_path):
    import tomllib

    config_file = tmp_path / "config.toml"
    config_file.write_text("not valid toml ][")
    with pytest.raises(tomllib.TOMLDecodeError):
        load_config(path=config_file)


def test_analysis_baseline_days_negative_raises(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[analysis]\nbaseline_days = -1\n")
    with pytest.raises(ValueError, match="baseline_days must be > 0"):
        load_config(path=config_file)


def test_analysis_equal_thresholds_raises(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "[analysis]\nthreshold_elevated = 2.0\nthreshold_unusual = 2.0\n"
    )
    with pytest.raises(ValueError, match="threshold_elevated.*must be less than"):
        load_config(path=config_file)


# ---------------------------------------------------------------------------
# [briefing] section
# ---------------------------------------------------------------------------


def test_briefing_defaults_when_no_file(tmp_path):
    cfg = load_config(path=tmp_path / "config.toml")
    assert cfg.briefing.headline_days == 7
    assert cfg.briefing.max_headlines == 50


def test_briefing_values_loaded_from_file(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[briefing]\nheadline_days = 14\nmax_headlines = 10\n")
    cfg = load_config(path=config_file)
    assert cfg.briefing.headline_days == 14
    assert cfg.briefing.max_headlines == 10


def test_briefing_headline_days_zero_raises(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[briefing]\nheadline_days = 0\n")
    with pytest.raises(ValueError, match="headline_days must be > 0"):
        load_config(path=config_file)


def test_briefing_max_headlines_zero_raises(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[briefing]\nmax_headlines = 0\n")
    with pytest.raises(ValueError, match="max_headlines must be > 0"):
        load_config(path=config_file)


def test_briefing_unrecognised_key_raises(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[briefing]\nunknown_key = 5\n")
    with pytest.raises(ValueError, match="unrecognised keys"):
        load_config(path=config_file)
