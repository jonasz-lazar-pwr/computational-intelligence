from src.core.config_service import ConfigService
from src.interfaces.core_interfaces import IConfigExpander, IConfigLoader, IConfigValidator


class DummyLoader(IConfigLoader):
    """Mock loader returning static data."""

    def __init__(self, data):
        self.data = data
        self.called = False

    def read(self):
        self.called = True
        return self.data


class DummyValidator(IConfigValidator):
    """Mock validator tracking validation calls."""

    def __init__(self):
        self.called = False

    def validate_root(self, data):
        self.called = True

    def validate_problem(self, problem):
        pass

    def validate_algorithm(self, algorithm, allow_lists):
        pass


class DummyExpander(IConfigExpander):
    """Mock expander recording input data."""

    def __init__(self, result=None):
        self.result = result or []
        self.called_with = None

    def expand(self, data):
        self.called_with = data
        return self.result


def test_load_all_returns_configs_and_logs(caplog):
    """Return expanded configs and log proper message."""
    dummy_data = {"experiments": [{"name": "test"}]}
    dummy_result = [{"name": "exp1"}]
    loader = DummyLoader(dummy_data)
    validator = DummyValidator()
    expander = DummyExpander(dummy_result)
    service = ConfigService(loader, validator, expander)

    with caplog.at_level("INFO"):
        configs = service.load_all()

    assert configs == dummy_result
    assert loader.called
    assert expander.called_with == dummy_data
    assert "Loaded 1 experiment configurations" in caplog.text


def test_singleton_behavior_between_instances():
    """Ensure ConfigService is a singleton."""
    l1, v1, e1 = DummyLoader({}), DummyValidator(), DummyExpander()
    l2, v2, e2 = DummyLoader({}), DummyValidator(), DummyExpander()
    s1 = ConfigService(l1, v1, e1)
    s2 = ConfigService(l2, v2, e2)
    assert s1 is s2
    assert isinstance(s1, ConfigService)
    assert hasattr(s1, "_configs")


def test_load_all_with_empty_result_logs_correctly(caplog):
    """Log zero configurations when expander returns empty list."""
    loader = DummyLoader({"sweep": []})
    validator = DummyValidator()
    expander = DummyExpander([])
    service = ConfigService(loader, validator, expander)

    with caplog.at_level("INFO"):
        result = service.load_all()

    assert result == []
    assert "Loaded 0 experiment configurations." in caplog.text


def test_load_all_stores_configs_for_later_access():
    """Update internal _configs state after load."""
    loader = DummyLoader({"experiments": [{"name": "a"}]})
    validator = DummyValidator()
    expander = DummyExpander([{"name": "exp_A"}])
    service = ConfigService(loader, validator, expander)
    result = service.load_all()
    assert service._configs == result
    assert isinstance(service._configs, list)
    assert service._configs[0]["name"] == "exp_A"
