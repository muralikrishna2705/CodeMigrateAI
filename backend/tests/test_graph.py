"""Tests for the LangGraph migration graph."""

from graph.conditions import complexity_condition, validate_condition


class TestComplexityCondition:
    def test_low_complexity_routes_to_retrieve(self):
        assert complexity_condition({"code_metrics": {"complexity": "low"}}) == "retrieve"

    def test_medium_complexity_routes_to_retrieve(self):
        assert (
            complexity_condition({"code_metrics": {"complexity": "medium"}})
            == "retrieve"
        )

    def test_high_complexity_routes_to_deep_analyze(self):
        assert (
            complexity_condition({"code_metrics": {"complexity": "high"}})
            == "deep_analyze"
        )

    def test_missing_metrics_defaults_to_low(self):
        assert complexity_condition({"code_metrics": None}) == "retrieve"

    def test_missing_complexity_key_defaults_to_low(self):
        assert complexity_condition({"code_metrics": {}}) == "retrieve"


class TestValidateCondition:
    def test_valid_returns_end(self):
        state = {
            "validation_result": {"valid": True},
            "retry_count": 0,
            "max_retries": 2,
        }
        assert validate_condition(state) == "end"

    def test_invalid_with_retries_returns_fix(self):
        state = {
            "validation_result": {"valid": False},
            "retry_count": 0,
            "max_retries": 2,
        }
        assert validate_condition(state) == "fix"

    def test_invalid_exhausted_returns_end(self):
        state = {
            "validation_result": {"valid": False},
            "retry_count": 2,
            "max_retries": 2,
        }
        assert validate_condition(state) == "end"

    def test_missing_validation_is_not_valid(self):
        state = {
            "validation_result": None,
            "retry_count": 0,
            "max_retries": 2,
        }
        assert validate_condition(state) == "fix"

    def test_partial_retry_still_routes_to_fix(self):
        state = {
            "validation_result": {"valid": False},
            "retry_count": 1,
            "max_retries": 2,
        }
        assert validate_condition(state) == "fix"


class TestGraphState:
    def test_graph_state_imports(self):
        from graph.state import GraphState

        # TypedDict — verifying the declared keys are present as annotations.
        for key in (
            "source_code",
            "source_language",
            "target_language",
            "retry_count",
            "max_retries",
        ):
            assert key in GraphState.__annotations__

    def test_migration_graph_compiles(self):
        from graph.migration_graph import build_migration_graph

        app = build_migration_graph()
        assert app is not None
        nodes = getattr(app, "nodes", {}) or {}
        for expected in ("analyze", "plan", "migrate", "validate", "fix"):
            assert expected in nodes, f"Missing node: {expected}"
