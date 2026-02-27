# driven-sub-ignition
0 PIC SIM
# Symplectic Differentiable PIC Reactor

A high-fidelity 0D-PIC engine designed to simulate driven-sub-ignition plasma dynamics. Built natively in PyTorch, this engine utilizes a **Symplectic Leapfrog (Verlet) Integrator** and an **Energy-Conserving Damping Gate** to ensure long-term stability and physical consistency in plasma heating regimes.

## 🔬 Core Physics
- **Symplectic Integration:** Employs a Kick-Drift-Kick Leapfrog scheme to conserve phase-space volume, eliminating the "numerical microwave" effect common in explicit Euler solvers.
- **MHD Stability:** Tracks the Kruskal-Shafranov limit via a dynamic Safety Factor ($q$) calculation, correctly identifying the transition from stable confinement to $m=1$ kink regimes.
- **Thermodynamic Governance:** Features a gated Langevin thermostat that enforces Maxwellian thermalization while allowing for adiabatic heating ramps during auxiliary power injection.
- **Driven Dynamics:** Models a Z-pinch geometry with a driven axial current ($I_z$), allowing for the study of self-generated azimuthal magnetic fields.

## 📊 Scientific Findings
Our research confirms that the reactor operates in a **driven sub-ignition state**. By implementing a 5% damping floor, we resolve the "Brownian runaway" artifact, showing that the plasma reaches a stable equilibrium during neutral beam injection (NBI) and transitions to a controlled cooldown upon driver removal. This accurately reflects the experimental reality of current tokamak devices (e.g., JET, ITER).

## 🚀 Installation
```bash
pip install torch numpy matplotlib
python run_reactor.py
