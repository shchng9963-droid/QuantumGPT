"""ChatLLM-NoTools baseline tests (v2.5 Sprint A, A3).

ChatLLM-NoTools is the *no-tool LLM control*: DeepSeek-Chat receives a
plain-text observation (health snapshot + drift signal + the catalog of
candidate actions) and must respond with one action name. No function-
calling. No tool execution. The chosen action's post-drift fidelity is
read directly from the oracle JSONL (same as Static-Pipeline).

What this baseline isolates:
  Agent − Static-Pipeline       = LLM + tools + planning + drift-aware
  Agent − ChatLLM-NoTools       = tools + planning  (LLM contribution removed)
  ChatLLM-NoTools − Static-Pipeline = pure LLM choice over the action grid

Tests use a mock provider object (no real API) so they're fast and CI-friendly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _oracle_row(**overrides):
    row = {
        "task_id": "FakeBrisbane:severe_sudden:ghz_5:seed0",
        "circuit": "ghz_5",
        "circuit_family": "hand_written",
        "backend": "FakeBrisbane",
        "profile": "severe_sudden",
        "seed": 0,
        "target_fidelity": 0.85,
        "pre_fidelity": 0.91,
        "first_post_drift_score": 0.70,
        "best_post_drift_score": 0.727,
        "final_score": 0.727,
        "post_drift_improvement": 0.027,
        "target_success": False,
        "oracle_feasible": False,
        "best_action": "zne_s1024",
        "actions": [
            {"action": "raw_s1024", "fidelity": 0.7020, "shots": 1024,
             "depth": 8, "metadata": {"kind": "raw"}},
            {"action": "transpile_o0_s1024", "fidelity": 0.6796, "shots": 1024,
             "depth": 8, "metadata": {"kind": "transpile", "opt_level": 0}},
            {"action": "transpile_o1_s1024", "fidelity": 0.6691, "shots": 1024,
             "depth": 8, "metadata": {"kind": "transpile", "opt_level": 1}},
            {"action": "transpile_o2_s1024", "fidelity": 0.6737, "shots": 1024,
             "depth": 8, "metadata": {"kind": "transpile", "opt_level": 2}},
            {"action": "transpile_o3_s1024", "fidelity": 0.6816, "shots": 1024,
             "depth": 8, "metadata": {"kind": "transpile", "opt_level": 3}},
            {"action": "zne_s1024", "fidelity": 0.727, "shots": 1024,
             "depth": None, "metadata": {"kind": "zne"}},
        ],
        "metadata": {"drift_score": 1.0, "avg_2q_error": 0.05, "avg_t1_us": 50.0},
    }
    row.update(overrides)
    return row


def _mock_chat_client(returned_action: str, prompt_tokens: int = 120, completion_tokens: int = 8):
    """Build a MagicMock OpenAI-compatible client that returns one fixed action."""
    client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = json.dumps({"action": returned_action})
    completion.usage = MagicMock(prompt_tokens=prompt_tokens,
                                  completion_tokens=completion_tokens,
                                  total_tokens=prompt_tokens + completion_tokens)
    client.chat.completions.create.return_value = completion
    return client


def test_chat_no_tools_calls_llm_once_and_uses_returned_action():
    from eval.baselines.chat_llm_notools import run_chat_llm_no_tools_task

    row = _oracle_row()
    client = _mock_chat_client("zne_s1024")
    result = run_chat_llm_no_tools_task(row, client=client, model="deepseek-chat")

    assert client.chat.completions.create.call_count == 1
    assert result.system == "ChatLLM-NoTools"
    assert result.provider == "deepseek"
    assert result.model == "deepseek-chat"
    # Picked action's fidelity, not oracle best
    assert result.first_post_fidelity == pytest.approx(0.727)
    assert result.best_post_fidelity == pytest.approx(0.727)
    # Tool-call list contains exactly the chosen action name
    assert result.tool_calls == ["zne_s1024"]
    # Cost is estimated from token usage at deepseek pricing
    cost = result.metadata["total_cost_usd"]
    assert cost > 0
    assert cost < 0.001  # 120in + 8out tokens at deepseek rates is well under $0.001


def test_chat_no_tools_records_token_usage_in_metadata():
    from eval.baselines.chat_llm_notools import run_chat_llm_no_tools_task

    row = _oracle_row()
    client = _mock_chat_client("transpile_o2_s1024", prompt_tokens=200, completion_tokens=20)
    result = run_chat_llm_no_tools_task(row, client=client)

    assert result.metadata["prompt_tokens"] == 200
    assert result.metadata["completion_tokens"] == 20
    assert result.metadata["total_tokens"] == 220
    assert result.metadata["execution_mode"] == "real"
    # Drift module not in play
    assert result.metadata["replan_triggered"] is False
    assert result.metadata["drift_alert_count"] == 0


def test_chat_no_tools_falls_back_to_static_action_on_unknown_response():
    """If the LLM returns a name not in the action grid, fall back to a fixed
    deterministic action and record the failure in metadata."""
    from eval.baselines.chat_llm_notools import run_chat_llm_no_tools_task

    row = _oracle_row()
    client = _mock_chat_client("nonsense_action")
    result = run_chat_llm_no_tools_task(row, client=client,
                                          fallback_action="transpile_o1_s1024")

    assert result.metadata["chosen_action"] == "transpile_o1_s1024"
    assert result.metadata["llm_returned_action"] == "nonsense_action"
    assert result.metadata["llm_action_invalid"] is True
    assert result.first_post_fidelity == pytest.approx(0.6691)
    # Tokens were still spent — cost is still recorded
    assert result.metadata["total_cost_usd"] > 0


def test_chat_no_tools_handles_malformed_json_gracefully():
    """Some LLM outputs may not be parseable JSON. The baseline must still
    produce a result row (using the fallback)."""
    from eval.baselines.chat_llm_notools import run_chat_llm_no_tools_task

    row = _oracle_row()
    client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = "I think we should use ZNE because... <not json>"
    completion.usage = MagicMock(prompt_tokens=100, completion_tokens=10, total_tokens=110)
    client.chat.completions.create.return_value = completion

    result = run_chat_llm_no_tools_task(row, client=client,
                                          fallback_action="transpile_o1_s1024")
    assert result.metadata["chosen_action"] == "transpile_o1_s1024"
    assert result.metadata["llm_action_invalid"] is True


def test_chat_no_tools_writes_paper_grade_jsonl(tmp_path):
    """Result must round-trip through write_public_agent_results unchanged."""
    from eval.baselines.chat_llm_notools import run_chat_llm_no_tools_task
    from eval.public_mqtbench_agent_eval import write_public_agent_results

    row = _oracle_row()
    client = _mock_chat_client("zne_s1024")
    result = run_chat_llm_no_tools_task(row, client=client)
    out = tmp_path / "chat_notools.jsonl"
    write_public_agent_results([result], out)

    written = json.loads(out.read_text())
    assert written["system"] == "ChatLLM-NoTools"
    assert written["provider"] == "deepseek"
    # Drift module never engages for chat-only baseline
    assert written["diagnostics"]["replan_triggered"] is False
    assert written["diagnostics"]["drift_alert_count"] == 0
    assert written["metrics"]["best_post_drift_score"] == pytest.approx(0.727)
