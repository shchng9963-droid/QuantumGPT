"""Day 14: Generate experiment protocol demo figures."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib
matplotlib.use('Agg')

from backends.dynamics_lab_adapter import DynamicsLabAdapter
from experiments import RabiExperiment, T1Experiment, RamseyExperiment

OUTPUT = Path(__file__).parent

def main():
    backend = DynamicsLabAdapter(qubit_freq_ghz=5.0, t1_us=200.0, t2_us=150.0)

    # 1. Rabi
    print("Generating Rabi figure...")
    rabi = RabiExperiment(amp_max=0.02, n_points=40, shots=2048)
    result = rabi.run(backend)
    result = rabi.analyze(result)
    fig = rabi.visualize(result, save_path=str(OUTPUT / "rabi_oscillation.png"))
    fig.savefig(str(OUTPUT / "rabi_oscillation.pdf"))
    print(f"  Saved: rabi_oscillation.png/pdf (R2={result.fit.r_squared:.4f})")

    pi_amp = result.fit.parameters['pi_amplitude']

    # 2. T1
    print("Generating T1 figure...")
    t1_exp = T1Experiment(delay_max_ns=600_000, n_points=35, pi_amplitude=pi_amp, shots=2048)
    t1_result = t1_exp.run(backend)
    t1_result = t1_exp.analyze(t1_result)
    fig = t1_exp.visualize(t1_result, save_path=str(OUTPUT / "t1_relaxation.png"))
    fig.savefig(str(OUTPUT / "t1_relaxation.pdf"))
    print(f"  Saved: t1_relaxation.png/pdf (T1={t1_result.fit.parameters['T1_us']:.1f}us)")

    # 3. Ramsey
    print("Generating Ramsey figure...")
    ramsey = RamseyExperiment(delay_max_ns=3000, n_points=50, pi_half_amplitude=pi_amp/2, shots=2048)
    ramsey_result = ramsey.run(backend)
    ramsey_result = ramsey.analyze(ramsey_result)
    fig = ramsey.visualize(ramsey_result, save_path=str(OUTPUT / "ramsey_fringes.png"))
    fig.savefig(str(OUTPUT / "ramsey_fringes.pdf"))
    print(f"  Saved: ramsey_fringes.png/pdf (T2*={ramsey_result.fit.parameters.get('T2_star_us', 'N/A')}us)")

    # 4. Combined figure
    print("Generating combined figure...")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Rabi subplot
    ax = axes[0]
    x, y = result.sweep_values * 1e3, result.measured_values
    ax.plot(x, y, 'o', ms=3, color='#2196F3')
    import numpy as np
    from experiments.rabi_experiment import _rabi_model
    p = result.fit.parameters
    xf = np.linspace(result.sweep_values[0], result.sweep_values[-1], 200) * 1e3
    yf = _rabi_model(xf/1e3, p['amplitude'], p['rabi_frequency'], p['phase'], p['offset'])
    ax.plot(xf, yf, '-', color='#F44336', lw=1.5)
    ax.set_xlabel('Amplitude (mV)')
    ax.set_ylabel('P(|1\u27e9)')
    ax.set_title(f'Rabi (R\u00b2={result.fit.r_squared:.3f})')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # T1 subplot
    ax = axes[1]
    x, y = t1_result.sweep_values / 1000, t1_result.measured_values
    ax.plot(x, y, 'o', ms=3, color='#009688')
    p = t1_result.fit.parameters
    xf = np.linspace(t1_result.sweep_values[0], t1_result.sweep_values[-1], 200)
    yf = p['amplitude'] * np.exp(-xf / p['T1_ns']) + p['offset']
    ax.plot(xf/1000, yf, '-', color='#E91E63', lw=1.5)
    ax.axvline(p['T1_us'], color='#FF9800', ls='--', lw=1)
    ax.set_xlabel('Delay (\u03bcs)')
    ax.set_ylabel('P(|1\u27e9)')
    ax.set_title(f'T1={p["T1_us"]:.0f}\u03bcs (R\u00b2={t1_result.fit.r_squared:.3f})')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Ramsey subplot
    ax = axes[2]
    x, y = ramsey_result.sweep_values / 1000, ramsey_result.measured_values
    ax.plot(x, y, 'o', ms=3, color='#673AB7')
    if ramsey_result.fit and ramsey_result.fit.r_squared > 0.5:
        p = ramsey_result.fit.parameters
        xf = np.linspace(ramsey_result.sweep_values[0], ramsey_result.sweep_values[-1], 500)
        f_det = p.get('detuning_mhz', 2.0) / 1000
        yf = p['amplitude'] * np.cos(2*np.pi*f_det*xf + p['phase']) * np.exp(-xf/p['T2_star_ns']) + p['offset']
        ax.plot(xf/1000, yf, '-', color='#FF5722', lw=1.5)
    ax.set_xlabel('Delay (\u03bcs)')
    ax.set_ylabel('P(|1\u27e9)')
    t2_str = f'{ramsey_result.fit.parameters.get("T2_star_us", 0):.1f}' if ramsey_result.fit else '?'
    ax.set_title(f'Ramsey T2*={t2_str}\u03bcs (R\u00b2={ramsey_result.fit.r_squared:.3f})')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(str(OUTPUT / "experiment_suite.png"), dpi=150, bbox_inches='tight')
    fig.savefig(str(OUTPUT / "experiment_suite.pdf"), bbox_inches='tight')
    print("  Saved: experiment_suite.png/pdf")
    plt.close('all')

    print("\nDone! All figures in demos/day14/")


if __name__ == "__main__":
    main()
