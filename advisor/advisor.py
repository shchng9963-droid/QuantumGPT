"""CalibrationAdvisor — autonomous multi-step calibration management chain.

Implements a 3-phase advisory loop:
  1. MONITOR  — collect health snapshots + run probe circuits over time
  2. DIAGNOSE — temporal reasoning over DuckDB history (trend detection,
                changepoint identification, qubit ranking)
  3. ACT      — generate concrete advisory actions with priorities

Actions the advisor can recommend:
  - EXCLUDE_QUBITS: remove specific qubits from routing
  - RECALIBRATE:    trigger recalibration (simulated)
  - REMAP_LAYOUT:   suggest a better qubit layout for a given circuit
  - APPLY_MITIGATION: recommend specific error mitigation strategies
  - WAIT:           drift is transient, wait for it to subside
  - OK:             device healthy, no action needed

Usage:
    from advisor.advisor import CalibrationAdvisor
    from backends.synthetic_drift import SyntheticDriftBackend, LINEAR_DECAY

    backend = SyntheticDriftBackend("FakeBrisbane", LINEAR_DECAY)
    advisor = CalibrationAdvisor(backend, db_path="demos/day8/advisor.duckdb")
    report = advisor.run_advisory_cycle(
        time_points=[0, 4, 8, 12],
        probe_circuits=["ghz_5", "qft_4"],
    )
    print(report.summary)
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np

from backends.base import ShadowBackend, BackendHealth
from tools.quantum_tools import ToolExecutor
from data.store import DataStore


# ═══════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════

class ActionType(str, Enum):
    EXCLUDE_QUBITS = "EXCLUDE_QUBITS"
    RECALIBRATE = "RECALIBRATE"
    REMAP_LAYOUT = "REMAP_LAYOUT"
    APPLY_MITIGATION = "APPLY_MITIGATION"
    WAIT = "WAIT"
    OK = "OK"


class Severity(str, Enum):
    NOMINAL = "nominal"
    WARNING = "warning"
    CRITICAL = "critical"


class TrendDirection(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    SUDDEN_CHANGE = "sudden_change"


@dataclass
class AdvisoryAction:
    """A single recommended action."""
    action: ActionType
    priority: int  # 1=highest, 5=lowest
    reason: str
    details: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Action({self.action.value}, P{self.priority}: {self.reason})"


@dataclass
class TemporalAnalysis:
    """Results of temporal analysis over health history."""
    trend: TrendDirection
    t1_slope: float  # us/hour (negative = degrading)
    t2_slope: float
    error_slope: float  # per hour (positive = degrading)
    drift_slope: float
    changepoints: list[float]  # time points where abrupt changes detected
    worst_qubits: list[int]  # qubits consistently underperforming
    best_qubits: list[int]  # qubits consistently strong
    fidelity_trend: Optional[TrendDirection] = None
    fidelity_slope: Optional[float] = None


@dataclass
class ProbeResult:
    """Result from a probe circuit run at a specific time."""
    time_hours: float
    circuit: str
    fidelity: float
    depth: int
    metadata: dict = field(default_factory=dict)


@dataclass
class AdvisoryReport:
    """Complete advisory report from one advisory cycle."""
    timestamp: str
    backend: str
    time_span_hours: float
    num_snapshots: int
    num_probes: int
    temporal_analysis: TemporalAnalysis
    actions: list[AdvisoryAction]
    health_timeline: list[dict]
    probe_results: list[ProbeResult]
    severity: Severity
    summary: str


# ═══════════════════════════════════════════════════════
# CalibrationAdvisor
# ═══════════════════════════════════════════════════════

class CalibrationAdvisor:
    """Autonomous multi-step calibration advisor.

    Runs a monitor→diagnose→act chain with temporal reasoning over
    DuckDB-persisted history.

    Args:
        backend: ShadowBackend instance (must support set_time() for
                 synthetic/replay backends)
        db_path: Path to DuckDB database for persistence
        qubit_sample_size: Number of qubits to sample for detailed analysis
    """

    # Thresholds
    DRIFT_WARNING = 0.2
    DRIFT_CRITICAL = 0.5
    FIDELITY_WARNING = 0.85
    FIDELITY_CRITICAL = 0.70
    T1_DEGRADATION_RATE = -5.0  # us/hour — worse than this triggers warning
    ERROR_GROWTH_RATE = 0.005   # per hour — worse than this triggers warning
    CHANGEPOINT_THRESHOLD = 0.15  # normalized jump to count as changepoint

    def __init__(
        self,
        backend: ShadowBackend,
        db_path: Optional[str] = None,
        qubit_sample_size: int = 20,
    ):
        self.backend = backend
        self.executor = ToolExecutor(backend)
        self.db = DataStore(db_path) if db_path else None
        self.qubit_sample_size = min(qubit_sample_size, backend.num_qubits)

        # Internal state
        self._health_timeline: list[dict] = []
        self._probe_results: list[ProbeResult] = []
        self._qubit_history: dict[int, list[dict]] = {}  # qubit -> [{t, t1, t2, re}]

    # ─── Phase 1: MONITOR ─────────────────────────────────

    def monitor(self, time_points: list[float],
                probe_circuits: Optional[list[str]] = None,
                shots: int = 4096) -> None:
        """Collect health snapshots and probe circuit results over time.

        Args:
            time_points: Simulated hours to sample (e.g. [0, 4, 8, 12])
            probe_circuits: Circuit names to run as probes at each time point
            shots: Shots per probe circuit
        """
        probe_circuits = probe_circuits or []

        for t in time_points:
            # Advance time on the backend
            if hasattr(self.backend, "set_time"):
                self.backend.set_time(t)

            # 1a. Health snapshot
            health = self.backend.get_health()
            health_dict = {
                "time_hours": t,
                "backend": health.name,
                "num_qubits": health.num_qubits,
                "avg_1q_error": health.avg_1q_error,
                "avg_2q_error": health.avg_2q_error,
                "avg_readout_error": health.avg_readout_error,
                "avg_t1_us": health.avg_t1_us,
                "avg_t2_us": health.avg_t2_us,
                "drift_score": health.drift_score,
                "calibration_age_minutes": health.calibration_age_minutes,
            }
            self._health_timeline.append(health_dict)

            # Persist to DuckDB
            if self.db:
                self.db.log_health(health_dict)

            # 1b. Sample qubit properties
            sample = list(range(self.qubit_sample_size))
            props = self.backend.get_qubit_properties(sample)
            for p in props:
                if p.qubit not in self._qubit_history:
                    self._qubit_history[p.qubit] = []
                self._qubit_history[p.qubit].append({
                    "time_hours": t,
                    "t1_us": p.t1_us,
                    "t2_us": p.t2_us,
                    "readout_error": p.readout_error,
                    "gate_error_1q": p.gate_errors.get("1q", 0),
                })

            if self.db:
                self.db.log_qubit_properties(
                    health.name,
                    [{"qubit": p.qubit, "t1_us": p.t1_us, "t2_us": p.t2_us,
                      "readout_error": p.readout_error, "gate_errors": p.gate_errors}
                     for p in props],
                )

            # 1c. Run probe circuits
            for circ_name in probe_circuits:
                result_str = self.executor.execute(
                    "run_circuit",
                    {"circuit_name": circ_name, "shots": shots},
                )
                try:
                    result = json.loads(result_str)
                    pr = ProbeResult(
                        time_hours=t,
                        circuit=circ_name,
                        fidelity=result.get("fidelity", 0),
                        depth=result.get("transpiled_depth", 0),
                        metadata=result.get("metadata", {}),
                    )
                    self._probe_results.append(pr)

                    if self.db:
                        self.db.log_circuit_run(result)
                except json.JSONDecodeError:
                    pass

    # ─── Phase 2: DIAGNOSE (temporal reasoning) ───────────

    def diagnose(self) -> TemporalAnalysis:
        """Analyze collected data for trends, changepoints, and qubit ranking.

        Returns:
            TemporalAnalysis with trend direction, slopes, changepoints,
            and qubit rankings.
        """
        if len(self._health_timeline) < 2:
            return TemporalAnalysis(
                trend=TrendDirection.STABLE,
                t1_slope=0, t2_slope=0, error_slope=0, drift_slope=0,
                changepoints=[], worst_qubits=[], best_qubits=[],
            )

        times = np.array([h["time_hours"] for h in self._health_timeline])
        t1s = np.array([h["avg_t1_us"] for h in self._health_timeline])
        t2s = np.array([h["avg_t2_us"] for h in self._health_timeline])
        errors = np.array([h["avg_2q_error"] for h in self._health_timeline])
        drifts = np.array([h["drift_score"] or 0 for h in self._health_timeline])

        # Linear regression for trends
        t1_slope = self._linear_slope(times, t1s)
        t2_slope = self._linear_slope(times, t2s)
        error_slope = self._linear_slope(times, errors)
        drift_slope = self._linear_slope(times, drifts)

        # Changepoint detection (simple: look for normalized jumps > threshold)
        changepoints = self._detect_changepoints(times, drifts)

        # Overall trend
        trend = self._classify_trend(t1_slope, error_slope, drift_slope, changepoints)

        # Qubit ranking
        worst, best = self._rank_qubits()

        # Fidelity trend (if we have probe data)
        fid_trend = None
        fid_slope = None
        if len(self._probe_results) >= 2:
            fid_times = np.array([pr.time_hours for pr in self._probe_results])
            fid_vals = np.array([pr.fidelity for pr in self._probe_results])
            fid_slope = self._linear_slope(fid_times, fid_vals)
            if fid_slope < -0.01:
                fid_trend = TrendDirection.DEGRADING
            elif fid_slope > 0.01:
                fid_trend = TrendDirection.IMPROVING
            else:
                fid_trend = TrendDirection.STABLE

        return TemporalAnalysis(
            trend=trend,
            t1_slope=t1_slope,
            t2_slope=t2_slope,
            error_slope=error_slope,
            drift_slope=drift_slope,
            changepoints=changepoints,
            worst_qubits=worst,
            best_qubits=best,
            fidelity_trend=fid_trend,
            fidelity_slope=fid_slope,
        )

    def _linear_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute slope via least-squares linear fit."""
        if len(x) < 2 or np.std(x) == 0:
            return 0.0
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0])

    def _detect_changepoints(self, times: np.ndarray,
                              values: np.ndarray) -> list[float]:
        """Simple changepoint detection: find time points where the
        normalized absolute difference between consecutive samples
        exceeds a threshold."""
        if len(values) < 2:
            return []
        val_range = max(np.ptp(values), 1e-6)
        diffs = np.abs(np.diff(values)) / val_range
        cps = []
        for i, d in enumerate(diffs):
            if d > self.CHANGEPOINT_THRESHOLD:
                cps.append(float(times[i + 1]))
        return cps

    def _classify_trend(self, t1_slope, error_slope, drift_slope,
                         changepoints) -> TrendDirection:
        """Classify overall health trend."""
        if changepoints:
            return TrendDirection.SUDDEN_CHANGE
        if t1_slope < self.T1_DEGRADATION_RATE or error_slope > self.ERROR_GROWTH_RATE:
            return TrendDirection.DEGRADING
        if t1_slope > abs(self.T1_DEGRADATION_RATE) * 0.5:
            return TrendDirection.IMPROVING
        return TrendDirection.STABLE

    def _rank_qubits(self) -> tuple[list[int], list[int]]:
        """Rank qubits by averaged quality score.

        Score = T1_norm * 0.4 + T2_norm * 0.3 + (1 - readout_err) * 0.3
        Returns (worst_5, best_5).
        """
        scores: dict[int, float] = {}
        for qubit, history in self._qubit_history.items():
            if not history:
                continue
            avg_t1 = np.mean([h["t1_us"] for h in history])
            avg_t2 = np.mean([h["t2_us"] for h in history])
            avg_re = np.mean([h["readout_error"] for h in history])
            # Simple combined score (higher = better)
            scores[qubit] = avg_t1 * 0.004 + avg_t2 * 0.003 + (1 - avg_re) * 0.3

        if not scores:
            return [], []

        sorted_q = sorted(scores.items(), key=lambda x: x[1])
        worst = [q for q, _ in sorted_q[:5]]
        best = [q for q, _ in sorted_q[-5:]]
        return worst, best

    # ─── Phase 3: ACT (generate advisory actions) ─────────

    def advise(self, analysis: TemporalAnalysis) -> tuple[list[AdvisoryAction], Severity]:
        """Generate concrete advisory actions from temporal analysis.

        Returns:
            (actions, overall_severity)
        """
        actions: list[AdvisoryAction] = []
        severity = Severity.NOMINAL

        latest = self._health_timeline[-1] if self._health_timeline else {}
        drift = latest.get("drift_score", 0)
        latest_fidelities = {}
        for pr in self._probe_results:
            latest_fidelities[pr.circuit] = pr.fidelity
        min_fidelity = min(latest_fidelities.values()) if latest_fidelities else 1.0

        # Rule 1: Changepoint detected → likely needs recalibration
        if analysis.changepoints:
            severity = Severity.CRITICAL
            actions.append(AdvisoryAction(
                action=ActionType.RECALIBRATE,
                priority=1,
                reason=(
                    f"Abrupt calibration change detected at t="
                    f"{', '.join(f'{cp:.1f}h' for cp in analysis.changepoints)}. "
                    f"Drift trend slope: {analysis.drift_slope:+.4f}/h."
                ),
                details={
                    "changepoints": analysis.changepoints,
                    "drift_slope": analysis.drift_slope,
                },
            ))

        # Rule 2: High drift score
        if drift > self.DRIFT_CRITICAL:
            severity = Severity.CRITICAL
            actions.append(AdvisoryAction(
                action=ActionType.RECALIBRATE,
                priority=1,
                reason=f"Drift score {drift:.3f} exceeds critical threshold ({self.DRIFT_CRITICAL}).",
                details={"drift_score": drift},
            ))
        elif drift > self.DRIFT_WARNING:
            if severity == Severity.NOMINAL:
                severity = Severity.WARNING
            actions.append(AdvisoryAction(
                action=ActionType.WAIT,
                priority=3,
                reason=(
                    f"Drift score {drift:.3f} is elevated (>{self.DRIFT_WARNING}) "
                    f"but below critical. Monitor for 1-2 hours."
                ),
                details={"drift_score": drift},
            ))

        # Rule 3: Degrading fidelity trend
        if analysis.fidelity_trend == TrendDirection.DEGRADING:
            if severity == Severity.NOMINAL:
                severity = Severity.WARNING
            actions.append(AdvisoryAction(
                action=ActionType.APPLY_MITIGATION,
                priority=2,
                reason=(
                    f"Fidelity is degrading at {analysis.fidelity_slope:+.4f}/h. "
                    f"Latest probe fidelities: "
                    + ", ".join(f"{c}={f:.3f}" for c, f in latest_fidelities.items())
                ),
                details={
                    "fidelity_slope": analysis.fidelity_slope,
                    "recommendation": "Apply ZNE or readout error mitigation",
                },
            ))

        # Rule 4: Low fidelity
        if min_fidelity < self.FIDELITY_CRITICAL:
            severity = Severity.CRITICAL
            actions.append(AdvisoryAction(
                action=ActionType.REMAP_LAYOUT,
                priority=1,
                reason=(
                    f"Minimum probe fidelity {min_fidelity:.3f} is below critical "
                    f"threshold ({self.FIDELITY_CRITICAL}). Remap to better qubits."
                ),
                details={
                    "best_qubits": analysis.best_qubits,
                    "worst_qubits": analysis.worst_qubits,
                },
            ))
        elif min_fidelity < self.FIDELITY_WARNING:
            if severity == Severity.NOMINAL:
                severity = Severity.WARNING
            actions.append(AdvisoryAction(
                action=ActionType.REMAP_LAYOUT,
                priority=2,
                reason=(
                    f"Minimum probe fidelity {min_fidelity:.3f} is moderate. "
                    f"Consider remapping to qubits {analysis.best_qubits}."
                ),
                details={"best_qubits": analysis.best_qubits},
            ))

        # Rule 5: Exclude consistently bad qubits
        if analysis.worst_qubits:
            # Check if worst qubits are significantly worse
            worst_scores = []
            best_scores = []
            for q in analysis.worst_qubits:
                if q in self._qubit_history and self._qubit_history[q]:
                    worst_scores.append(np.mean([h["t1_us"] for h in self._qubit_history[q]]))
            for q in analysis.best_qubits:
                if q in self._qubit_history and self._qubit_history[q]:
                    best_scores.append(np.mean([h["t1_us"] for h in self._qubit_history[q]]))

            if worst_scores and best_scores:
                ratio = np.mean(worst_scores) / max(np.mean(best_scores), 1e-6)
                if ratio < 0.5:  # worst qubits have < 50% of best T1
                    actions.append(AdvisoryAction(
                        action=ActionType.EXCLUDE_QUBITS,
                        priority=2,
                        reason=(
                            f"Qubits {analysis.worst_qubits} have avg T1 "
                            f"{np.mean(worst_scores):.1f}us vs best "
                            f"{np.mean(best_scores):.1f}us ({ratio:.0%}). "
                            f"Exclude from routing."
                        ),
                        details={
                            "qubits": analysis.worst_qubits,
                            "t1_ratio": ratio,
                        },
                    ))

        # Rule 6: Steady degradation (T1 slope)
        if analysis.t1_slope < self.T1_DEGRADATION_RATE:
            if severity == Severity.NOMINAL:
                severity = Severity.WARNING
            actions.append(AdvisoryAction(
                action=ActionType.RECALIBRATE,
                priority=2,
                reason=(
                    f"T1 degrading at {analysis.t1_slope:+.2f} us/h. "
                    f"At this rate, device will become unusable in "
                    f"{abs(latest.get('avg_t1_us', 100) / min(analysis.t1_slope, -0.01)):.0f}h."
                ),
                details={"t1_slope": analysis.t1_slope},
            ))

        # Rule 7: All good
        if not actions:
            actions.append(AdvisoryAction(
                action=ActionType.OK,
                priority=5,
                reason=(
                    f"Device healthy. Drift={drift:.3f}, "
                    f"T1 trend={analysis.t1_slope:+.2f}us/h, "
                    f"fidelity stable."
                ),
            ))

        # Sort by priority
        actions.sort(key=lambda a: a.priority)
        return actions, severity

    # ─── Full advisory cycle ──────────────────────────────

    def run_advisory_cycle(
        self,
        time_points: list[float],
        probe_circuits: Optional[list[str]] = None,
        shots: int = 4096,
    ) -> AdvisoryReport:
        """Run the full monitor→diagnose→act cycle.

        Args:
            time_points: Simulated hours to monitor (e.g. [0, 4, 8, 12])
            probe_circuits: Circuits to run as probes
            shots: Shots per probe

        Returns:
            AdvisoryReport with full analysis and recommended actions.
        """
        # Phase 1: Monitor
        self.monitor(time_points, probe_circuits, shots)

        # Phase 2: Diagnose
        analysis = self.diagnose()

        # Phase 3: Act
        actions, severity = self.advise(analysis)

        # Generate summary
        summary = self._generate_summary(analysis, actions, severity)

        return AdvisoryReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            backend=self.backend.name,
            time_span_hours=max(time_points) - min(time_points) if time_points else 0,
            num_snapshots=len(self._health_timeline),
            num_probes=len(self._probe_results),
            temporal_analysis=analysis,
            actions=actions,
            health_timeline=self._health_timeline,
            probe_results=self._probe_results,
            severity=severity,
            summary=summary,
        )

    def _generate_summary(self, analysis: TemporalAnalysis,
                           actions: list[AdvisoryAction],
                           severity: Severity) -> str:
        """Generate a human-readable summary of the advisory report."""
        lines = []
        lines.append(f"=== Calibration Advisory Report ===")
        lines.append(f"Backend: {self.backend.name}")
        lines.append(f"Severity: {severity.value.upper()}")
        lines.append(f"Overall trend: {analysis.trend.value}")
        lines.append("")

        # Health summary
        if self._health_timeline:
            first = self._health_timeline[0]
            last = self._health_timeline[-1]
            lines.append(f"Health progression (t={first['time_hours']:.0f}h -> t={last['time_hours']:.0f}h):")
            lines.append(f"  T1:     {first['avg_t1_us']:.1f} -> {last['avg_t1_us']:.1f} us "
                         f"(slope: {analysis.t1_slope:+.2f}/h)")
            lines.append(f"  T2:     {first['avg_t2_us']:.1f} -> {last['avg_t2_us']:.1f} us "
                         f"(slope: {analysis.t2_slope:+.2f}/h)")
            lines.append(f"  2Q err: {first['avg_2q_error']:.5f} -> {last['avg_2q_error']:.5f} "
                         f"(slope: {analysis.error_slope:+.6f}/h)")
            lines.append(f"  Drift:  {first.get('drift_score', 0):.3f} -> {last.get('drift_score', 0):.3f}")
            lines.append("")

        # Changepoints
        if analysis.changepoints:
            lines.append(f"Changepoints detected at: "
                         + ", ".join(f"t={cp:.1f}h" for cp in analysis.changepoints))
            lines.append("")

        # Fidelity
        if self._probe_results:
            lines.append("Probe circuit results:")
            by_circuit: dict[str, list[ProbeResult]] = {}
            for pr in self._probe_results:
                by_circuit.setdefault(pr.circuit, []).append(pr)
            for circ, prs in by_circuit.items():
                fids = [f"{pr.fidelity:.4f}@t={pr.time_hours:.0f}h" for pr in prs]
                lines.append(f"  {circ}: {', '.join(fids)}")
            if analysis.fidelity_slope is not None:
                lines.append(f"  Fidelity trend: {analysis.fidelity_trend.value} "
                             f"({analysis.fidelity_slope:+.4f}/h)")
            lines.append("")

        # Qubit analysis
        if analysis.worst_qubits:
            lines.append(f"Worst qubits (by T1/T2/readout composite): {analysis.worst_qubits}")
            lines.append(f"Best qubits: {analysis.best_qubits}")
            lines.append("")

        # Actions
        lines.append(f"Recommended actions ({len(actions)}):")
        for i, action in enumerate(actions, 1):
            lines.append(f"  {i}. [{action.action.value}] P{action.priority} — {action.reason}")
            if action.details:
                for k, v in action.details.items():
                    lines.append(f"     {k}: {v}")
        lines.append("")

        return "\n".join(lines)

    # ─── DuckDB temporal queries ──────────────────────────

    def query_health_trend(self, hours_back: float = 24) -> list[dict]:
        """Query health history from DuckDB for temporal analysis."""
        if self.db is None:
            return self._health_timeline
        return self.db.query(
            f"SELECT * FROM health_snapshots "
            f"WHERE ts >= now() - INTERVAL '{int(hours_back)} hours' "
            f"ORDER BY ts",
        )

    def query_fidelity_trend(self, circuit: str = None) -> list[dict]:
        """Query fidelity history from DuckDB."""
        if self.db is None:
            return [{"circuit": pr.circuit, "fidelity": pr.fidelity,
                      "time_hours": pr.time_hours}
                    for pr in self._probe_results
                    if circuit is None or pr.circuit == circuit]
        if circuit:
            return self.db.fidelity_trend(circuit)
        return self.db.fidelity_trend()

    def query_worst_qubits(self, top_n: int = 5) -> list[dict]:
        """Query worst qubits from DuckDB."""
        if self.db is None:
            return []
        return self.db.worst_qubits(self.backend.name, top_n)
