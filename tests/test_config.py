import pytest

from financial_news.config import Config, LoggingConfig, load_config


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
