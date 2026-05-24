"""QuantumGPT Web API -- FastAPI application."""

from __future__ import annotations
import asyncio, time
from contextlib import asynccontextmanager
from typing import Any
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backends.dynamics_lab_adapter import DynamicsLabAdapter
from backends.lab_backend import DeviceSnapshot
from experiments import RabiExperiment, RamseyExperiment, T1Experiment

_backends: dict[str, DynamicsLabAdapter] = {}
_experiment_history: list[dict] = []

def _get_or_create_backend(name: str = "DynamicsSim") -> DynamicsLabAdapter:
    if name not in _backends:
        _backends[name] = DynamicsLabAdapter(qubit_freq_ghz=5.0, t1_us=200.0, t2_us=150.0, backend_name=name)
    return _backends[name]

@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_or_create_backend("DynamicsSim")
    yield
    _backends.clear()

app = FastAPI(title="QuantumGPT API", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class BackendInfo(BaseModel):
    name: str; num_qubits: int; device_type: str; supports_pulse: bool
    qubit_frequencies_ghz: list[float]; t1_us: list[float]; t2_us: list[float]
    readout_errors: list[float]; calibration_age_minutes: float

class ExperimentConfig(BaseModel):
    protocol: str = Field(..., description="rabi, ramsey, t1")
    qubit: int = 0; n_points: int = 30; shots: int = 1024
    amp_min: float = 0.0; amp_max: float = 0.08; pulse_duration_ns: float = 100.0
    delay_max_ns: float = 600_000.0; pi_amplitude: float | None = None
    ramsey_delay_max_ns: float = 3000.0; artificial_detuning_mhz: float = 2.0

class ExperimentResponse(BaseModel):
    protocol: str; sweep_parameter: str
    sweep_values: list[float]; measured_values: list[float]
    measured_errors: list[float] | None = None
    fit_parameters: dict[str, float] | None = None
    fit_uncertainties: dict[str, float] | None = None
    r_squared: float | None = None; model: str | None = None
    metadata: dict[str, Any] = {}; elapsed_seconds: float = 0.0

class HealthResponse(BaseModel):
    status: str; backends: list[BackendInfo]

@app.get("/api/health", response_model=HealthResponse)
async def get_health():
    backends = []
    for name, be in _backends.items():
        state = be.get_device_state()
        backends.append(BackendInfo(name=state.backend_name, num_qubits=state.num_qubits,
            device_type=be.device_type, supports_pulse=be.supports_pulse,
            qubit_frequencies_ghz=state.qubit_frequencies_ghz, t1_us=state.t1_us,
            t2_us=state.t2_us, readout_errors=state.readout_errors,
            calibration_age_minutes=state.calibration_age_minutes))
    status = "healthy"
    for b in backends:
        if any(t < 50 for t in b.t1_us): status = "critical"; break
        if any(t < 100 for t in b.t1_us): status = "warning"
    return HealthResponse(status=status, backends=backends)

@app.get("/api/backends")
async def list_backends():
    return [{"name": n, "num_qubits": b.num_qubits, "supports_pulse": b.supports_pulse} for n, b in _backends.items()]

@app.get("/api/backends/{name}/state")
async def get_backend_state(name: str):
    if name not in _backends: raise HTTPException(404, f"Backend not found: {name}")
    s = _backends[name].get_device_state()
    return {"backend_name": s.backend_name, "num_qubits": s.num_qubits,
        "qubit_frequencies_ghz": s.qubit_frequencies_ghz, "t1_us": s.t1_us,
        "t2_us": s.t2_us, "readout_errors": s.readout_errors,
        "gate_errors_1q": s.gate_errors_1q, "calibration_age_minutes": s.calibration_age_minutes}

@app.post("/api/experiments/run", response_model=ExperimentResponse)
async def run_experiment(config: ExperimentConfig):
    backend = _get_or_create_backend("DynamicsSim")
    t0 = time.time()
    if config.protocol == "rabi":
        exp = RabiExperiment(qubit=config.qubit, amp_min=config.amp_min, amp_max=config.amp_max,
            n_points=config.n_points, pulse_duration_ns=config.pulse_duration_ns, shots=config.shots)
    elif config.protocol == "t1":
        exp = T1Experiment(qubit=config.qubit, delay_max_ns=config.delay_max_ns, n_points=config.n_points,
            pi_amplitude=config.pi_amplitude or 0.005, pi_duration_ns=config.pulse_duration_ns, shots=config.shots)
    elif config.protocol == "ramsey":
        pi_amp = config.pi_amplitude or 0.005
        exp = RamseyExperiment(qubit=config.qubit, delay_max_ns=config.ramsey_delay_max_ns,
            n_points=config.n_points, pi_half_amplitude=pi_amp/2,
            pi_half_duration_ns=config.pulse_duration_ns,
            artificial_detuning_mhz=config.artificial_detuning_mhz, shots=config.shots)
    else:
        raise HTTPException(400, f"Unknown protocol: {config.protocol}")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, exp.run, backend)
    result = await loop.run_in_executor(None, exp.analyze, result)
    elapsed = time.time() - t0
    resp = ExperimentResponse(protocol=result.protocol_name, sweep_parameter=result.sweep_parameter,
        sweep_values=result.sweep_values.tolist(), measured_values=result.measured_values.tolist(),
        measured_errors=result.measured_errors.tolist() if result.measured_errors is not None else None,
        fit_parameters=result.fit.parameters if result.fit else None,
        fit_uncertainties=result.fit.uncertainties if result.fit else None,
        r_squared=result.fit.r_squared if result.fit else None,
        model=result.fit.model if result.fit else None,
        metadata=result.metadata, elapsed_seconds=elapsed)
    _experiment_history.append({"timestamp": time.time(), "protocol": config.protocol,
        "r_squared": result.fit.r_squared if result.fit else 0,
        "summary": result.summary() if result.fit else "no fit"})
    return resp

@app.get("/api/experiments/history")
async def get_experiment_history(limit: int = 20):
    return _experiment_history[-limit:]

@app.get("/api/experiments/protocols")
async def list_protocols():
    return [
        {"name": "rabi", "description": "Rabi oscillation -- calibrate pi-pulse amplitude"},
        {"name": "t1", "description": "T1 relaxation -- measure energy decay time"},
        {"name": "ramsey", "description": "Ramsey fringes -- measure T2* dephasing time"},
    ]

@app.websocket("/ws/telemetry")
async def telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    backend = _get_or_create_backend("DynamicsSim")
    try:
        while True:
            state = backend.get_device_state()
            await websocket.send_json({"timestamp": time.time(), "t1_us": state.t1_us,
                "t2_us": state.t2_us, "readout_errors": state.readout_errors})
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
