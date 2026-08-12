"""Consistency checks for the canonical runtime tool registry."""

from agent.state import ArtifactType
from tools.quantum_tools import TOOL_DEFINITIONS, ToolExecutor
from tools.registry import TOOL_RUNTIME_SPECS, validate_tool_registry


def test_runtime_registry_matches_llm_tool_surface():
    validate_tool_registry(TOOL_DEFINITIONS)
    assert set(TOOL_RUNTIME_SPECS) == {item["name"] for item in TOOL_DEFINITIONS}


def test_runtime_handlers_exist_on_executor():
    for spec in TOOL_RUNTIME_SPECS.values():
        assert hasattr(ToolExecutor, spec.handler_name)


def test_registered_artifact_types_are_valid():
    valid = {item.value for item in ArtifactType}
    for spec in TOOL_RUNTIME_SPECS.values():
        assert spec.artifact_type is None or spec.artifact_type in valid
