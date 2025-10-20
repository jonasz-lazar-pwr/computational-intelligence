from pathlib import Path

import pytest
import yaml

from src.core.config_loader import ConfigLoader
from src.interfaces.core_interfaces import IConfigValidator


class DummyValidator(IConfigValidator):
    """Mock validator tracking validate_root calls."""

    def __init__(self):
        self.called_with = None

    def validate_root(self, data):
        self.called_with = data

    def validate_problem(self, problem):
        pass

    def validate_algorithm(self, algorithm, allow_lists):
        pass


@pytest.fixture
def tmp_yaml_file(tmp_path: Path) -> Path:
    """Create temporary YAML file with valid content."""
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"experiments": [{"name": "exp1"}]}, f)
    return path


@pytest.fixture
def validator():
    return DummyValidator()


def test_read_loads_yaml_and_validates(tmp_yaml_file, validator):
    """Load YAML correctly and call validator."""
    loader = ConfigLoader(str(tmp_yaml_file), validator)
    data = loader.read()
    assert isinstance(data, dict)
    assert validator.called_with == data
    assert "config.yaml" in str(loader._path)


def test_read_raises_file_not_found(tmp_path, validator):
    """Raise FileNotFoundError for missing YAML file."""
    missing = tmp_path / "missing.yaml"
    loader = ConfigLoader(str(missing), validator)
    with pytest.raises(FileNotFoundError):
        loader.read()


def test_read_with_empty_yaml_file(tmp_path, validator):
    """Handle empty YAML file gracefully."""
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    loader = ConfigLoader(str(path), validator)
    data = loader.read()
    assert data is None
    assert validator.called_with is None


def test_read_with_invalid_yaml_syntax(tmp_path, validator):
    """Raise YAMLError for invalid syntax."""
    path = tmp_path / "broken.yaml"
    path.write_text("{ invalid: [unterminated", encoding="utf-8")
    loader = ConfigLoader(str(path), validator)
    with pytest.raises(yaml.YAMLError):
        loader.read()


def test_logger_debug_called_during_read(tmp_yaml_file, validator, caplog):
    """Log debug message during config read."""
    loader = ConfigLoader(str(tmp_yaml_file), validator)
    with caplog.at_level("DEBUG"):
        loader.read()
    assert "Loading config file" in caplog.text
