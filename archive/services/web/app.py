"""QuantumGPT Lab Console -- Streamlit Web Dashboard.

Run with: streamlit run web/app.py --server.port 8501
"""

import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backends.dynamics_lab_adapter import DynamicsLabAdapter
from experiments import RabiExperiment, RamseyExperiment, T1Experiment

# --- Page config ---
st.set_page_config(
    page_title="QuantumGPT Lab Console",
    page_icon="\u269b",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    .metric-card {
        background: rgba(30,41,59,0.8);
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #22c55e;
    }
    .metric-card.warn { border-left-color: #eab308; }
    h1, h2, h3 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# --- Backend singleton ---
@st.cache_resource
def get_backend():
    return DynamicsLabAdapter(
        qubit_freq_ghz=5.0,
        t1_us=200.0,
        t2_us=150.0,
        backend_name="DynamicsSim",
    )


# --- Sidebar ---
st.sidebar.markdown("## \u269b QuantumGPT Lab Console")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Experiment Lab", "Device Monitor", "History"],
    index=0,
)

backend = get_backend()
state = backend.get_device_state()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# --- Dashboard Page ---
if page == "Dashboard":
    st.title("\u269b QuantumGPT Lab Console")
    st.caption("Pulse-level quantum hardware characterization")

    # Status badge
    col_status, col_spacer = st.columns([1, 4])
    with col_status:
        st.success("\u2705 System Healthy")

    st.markdown("---")

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Qubits", state.num_qubits)
    c2.metric("T1", f"{state.t1_us[0]:.0f} \u03bcs")
    c3.metric("T2", f"{state.t2_us[0]:.0f} \u03bcs")
    c4.metric("Readout Error", f"{state.readout_errors[0]*100:.2f}%")
    c5.metric("Frequency", f"{state.qubit_frequencies_ghz[0]:.3f} GHz")

    st.markdown("---")

    # Quick experiment
    st.subheader("Quick Experiment")
    col_exp, col_res = st.columns(2)

    with col_exp:
        quick_proto = st.selectbox("Protocol", ["rabi", "t1", "ramsey"], key="quick_proto")
        if st.button("Run Quick Experiment", type="primary", use_container_width=True):
            with st.spinner("Running..."):
                t0 = time.time()
                if quick_proto == "rabi":
                    exp = RabiExperiment(n_points=30, shots=1024)
                elif quick_proto == "t1":
                    exp = T1Experiment(n_points=25, pi_amplitude=0.005, shots=1024)
                else:
                    exp = RamseyExperiment(n_points=30, pi_half_amplitude=0.0025, shots=1024)

                result = exp.run(backend)
                result = exp.analyze(result)
                elapsed = time.time() - t0

                st.session_state.last_result = result
                st.session_state.history.append({
                    "protocol": quick_proto,
                    "r_squared": result.fit.r_squared if result.fit else 0,
                    "time": time.strftime("%H:%M:%S"),
                    "elapsed": elapsed,
                    "params": result.fit.parameters if result.fit else {},
                })
                st.rerun()

    with col_res:
        result = st.session_state.last_result
        if result and result.fit:
            st.markdown(f"**{result.protocol_name.upper()}** | R\u00b2 = {result.fit.r_squared:.4f}")
            for k, v in result.fit.parameters.items():
                st.text(f"  {k}: {v:.6g}")


# --- Experiment Lab Page ---
elif page == "Experiment Lab":
    st.title("\u2699 Experiment Lab")

    col_config, col_result = st.columns([1, 2])

    with col_config:
        st.subheader("Configuration")
        protocol = st.selectbox("Protocol", ["rabi", "t1", "ramsey"])
        n_points = st.slider("Sweep Points", 10, 80, 30)
        shots = st.select_slider("Shots per Point", [256, 512, 1024, 2048, 4096], value=1024)

        st.markdown("---")

        if protocol == "rabi":
            amp_max = st.slider("Max Amplitude", 0.01, 0.2, 0.08, 0.01)
            pulse_dur = st.slider("Pulse Duration (ns)", 20, 500, 100, 10)
        elif protocol == "t1":
            delay_max = st.slider("Max Delay (\u03bcs)", 100, 1000, 600, 50)
            pi_amp = st.number_input("Pi Amplitude", value=0.005, format="%.4f")
        else:
            delay_max_r = st.slider("Max Delay (ns)", 500, 10000, 3000, 100)
            detuning = st.slider("Artificial Detuning (MHz)", 0.5, 10.0, 2.0, 0.5)

        run_btn = st.button("Run Experiment", type="primary", use_container_width=True)

    with col_result:
        if run_btn:
            with st.spinner(f"Running {protocol}..."):
                t0 = time.time()

                if protocol == "rabi":
                    exp = RabiExperiment(amp_max=amp_max, n_points=n_points,
                                        pulse_duration_ns=pulse_dur, shots=shots)
                elif protocol == "t1":
                    exp = T1Experiment(delay_max_ns=delay_max*1000, n_points=n_points,
                                      pi_amplitude=pi_amp, shots=shots)
                else:
                    exp = RamseyExperiment(delay_max_ns=delay_max_r, n_points=n_points,
                                          pi_half_amplitude=0.0025,
                                          artificial_detuning_mhz=detuning, shots=shots)

                result = exp.run(backend)
                result = exp.analyze(result)
                elapsed = time.time() - t0

                st.session_state.last_result = result
                st.session_state.history.append({
                    "protocol": protocol,
                    "r_squared": result.fit.r_squared if result.fit else 0,
                    "time": time.strftime("%H:%M:%S"),
                    "elapsed": elapsed,
                    "params": result.fit.parameters if result.fit else {},
                })

        result = st.session_state.last_result
        if result:
            st.subheader(f"Result: {result.protocol_name.upper()}")

            # Plot with Plotly
            fig = go.Figure()

            # Data points
            fig.add_trace(go.Scatter(
                x=result.sweep_values,
                y=result.measured_values,
                mode='markers',
                name='Data',
                marker=dict(color='#3b82f6', size=6),
                error_y=dict(
                    type='data',
                    array=result.measured_errors if result.measured_errors is not None else None,
                    visible=result.measured_errors is not None,
                    color='rgba(59,130,246,0.3)',
                ),
            ))

            # Fit curve
            if result.fit and result.fit.r_squared > 0.5:
                p = result.fit.parameters
                x_fine = np.linspace(result.sweep_values[0], result.sweep_values[-1], 200)

                if result.protocol_name == "rabi":
                    y_fit = p['amplitude'] * np.cos(2*np.pi*p['rabi_frequency']*x_fine + p['phase']) + p['offset']
                elif result.protocol_name == "t1":
                    y_fit = p['amplitude'] * np.exp(-x_fine / p['T1_ns']) + p['offset']
                elif result.protocol_name == "ramsey":
                    f_det = p.get('detuning_mhz', 2.0) / 1000
                    y_fit = p['amplitude'] * np.cos(2*np.pi*f_det*x_fine + p['phase']) * np.exp(-x_fine / p['T2_star_ns']) + p['offset']
                else:
                    y_fit = None

                if y_fit is not None:
                    fig.add_trace(go.Scatter(
                        x=x_fine, y=y_fit,
                        mode='lines', name='Fit',
                        line=dict(color='#ef4444', width=2),
                    ))

            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,23,42,0.8)',
                xaxis_title=result.sweep_parameter,
                yaxis_title='P(|1\u27e9)',
                yaxis_range=[-0.05, 1.05],
                height=400,
                margin=dict(l=50, r=20, t=40, b=50),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Fit parameters
            if result.fit:
                st.markdown(f"**R\u00b2 = {result.fit.r_squared:.4f}** | Model: `{result.fit.model}`")
                param_cols = st.columns(min(len(result.fit.parameters), 4))
                for i, (k, v) in enumerate(result.fit.parameters.items()):
                    param_cols[i % len(param_cols)].metric(k, f"{v:.4g}")
        else:
            st.info("Configure and run an experiment to see results.")


# --- Device Monitor Page ---
elif page == "Device Monitor":
    st.title("\U0001f4e1 Device Monitor")

    state = backend.get_device_state()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Device Properties")
        st.json({
            "backend": state.backend_name,
            "num_qubits": state.num_qubits,
            "device_type": backend.device_type,
            "supports_pulse": backend.supports_pulse,
            "qubit_frequency_ghz": state.qubit_frequencies_ghz[0],
            "T1_us": state.t1_us[0],
            "T2_us": state.t2_us[0],
            "readout_error": state.readout_errors[0],
            "calibration_age_min": round(state.calibration_age_minutes, 2),
        })

    with col2:
        st.subheader("Calibration Store")
        if state.extra and "calibration_store" in state.extra:
            st.json(state.extra["calibration_store"])
        else:
            st.info("No calibration data yet. Run a Rabi experiment to calibrate.")


# --- History Page ---
elif page == "History":
    st.title("\U0001f4ca Experiment History")

    if not st.session_state.history:
        st.info("No experiments run yet. Go to Experiment Lab to start.")
    else:
        # Summary metrics
        total = len(st.session_state.history)
        avg_r2 = np.mean([h['r_squared'] for h in st.session_state.history])
        st.metric("Total Experiments", total)
        st.metric("Average R\u00b2", f"{avg_r2:.4f}")

        st.markdown("---")

        # Table
        for i, h in enumerate(reversed(st.session_state.history)):
            with st.expander(f"#{total-i} | {h['protocol'].upper()} | R\u00b2={h['r_squared']:.4f} | {h['time']}"):
                st.text(f"Elapsed: {h['elapsed']:.2f}s")
                if h['params']:
                    for k, v in h['params'].items():
                        st.text(f"  {k}: {v:.6g}")
