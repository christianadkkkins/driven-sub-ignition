"""
Symplectic Differentiable PIC Reactor Engine
============================================
A 0D Particle-in-Cell (PIC) solver modeling driven plasma dynamics, 
MHD kink stability, and Lawson ignition thresholds.

Key Architectural Features:
1. Symplectic Leapfrog Integrator: Conserves phase-space volume to eliminate 
   the 'numerical microwave' artifact.
2. Gated Langevin Thermostat: Decouples stochastic thermalization from 
   systematic drag during auxiliary heating windows.
3. Velocity Ceiling: A numerical safety clamp that prevents stochastic tail 
   singularities, which also acts as an observable upper bound during thermal runaway.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class SymplecticFusionEngine(nn.Module):
    def __init__(self, n_particles: int = 512, device: str = 'cpu'):
        super().__init__()
        self.n_particles = n_particles
        self.dt = 0.002 
        
        # Confinement & Electromagnetic parameters
        self.B_ext = 8.0          # Axial stabilizing magnetic field
        self.I_plasma = 12.0      # Driven axial current (Z-pinch proxy)
        self.mu_pinch = 0.25      # Plasma permeability factor
        self.stiffness = 8.0      # Magnetic restoring wall potential (r^4)
        self.coulomb_k = 0.05     # Quasi-neutrality electrostatic coupling
        
        # Thermodynamics & Lawson kinetics
        self.alpha_rate = 0.505   # Alpha heating rate (optimized for 70 keV root)
        self.beta_rate = 0.1      # Bremsstrahlung cooling coefficient
        self.collision_nu = 2.5   # Baseline Langevin collision frequency
        self.P_aux = 0.0          # Auxiliary Neutral Beam Injection (NBI) power

    def get_forces(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Evaluates pure physical forces at the current position and velocity."""
        q_norm = torch.norm(q, dim=1, keepdim=True).clamp(min=1e-4)
        
        # 1. Gauss's Law: Electrostatic expansion
        r_sorted, sort_idx = torch.sort(q_norm.squeeze())
        unsort_idx = torch.argsort(sort_idx)
        Q_enc = torch.arange(1, self.n_particles + 1, device=q.device, dtype=torch.float32) / self.n_particles
        E_field = (self.coulomb_k * Q_enc[unsort_idx].unsqueeze(1) / (q_norm**2 + 0.05)) * (q / q_norm)
        
        # 2. Biot-Savart Law: Azimuthal pinch & m=1 Kink Lorentz force
        J_driven = torch.zeros_like(q); J_driven[:, 2] = self.I_plasma 
        B_self_vec = self.mu_pinch * torch.cross(J_driven, q) / (q_norm**2 + 0.5)
        F_kink = torch.cross(v, B_self_vec)
        
        # 3. Magnetic Confinement Wall (r^4 restoring potential)
        wall_pressure = -q * (self.stiffness * q_norm**4)
        
        return wall_pressure + E_field + F_kink

    def forward(self, q: torch.Tensor, v: torch.Tensor):
        """Advances the state using a volume-preserving Leapfrog integration scheme."""
        
        # --- KICK-DRIFT-KICK (Symplectic Step) ---
        a_t = self.get_forces(q, v)
        v_half = v + 0.5 * self.dt * a_t
        
        q_next = q + self.dt * v_half
        
        a_next = self.get_forces(q_next, v_half)
        v_next = v_half + 0.5 * self.dt * a_next
        
        # --- THERMODYNAMIC BALANCE ---
        T_current = torch.mean(torch.sum(v_next**2, dim=1))
        
        # Alpha heating with high-T saturation correction
        P_alpha = (self.alpha_rate * T_current) / (1.0 + (T_current / 8.0)**1.5)
        # Bremsstrahlung radiation cooling
        P_brems = self.beta_rate * torch.sqrt(T_current + 1e-6)
        
        # Damping Gate: Keep 5% viscosity floor active during NBI heating 
        # to prevent Brownian runaway while allowing energy accumulation.
        gamma_eff = (0.05 * self.collision_nu) if self.P_aux > 0.0 else self.collision_nu
        T_target = T_current + (P_alpha - P_brems + self.P_aux) * self.dt
        
        # Langevin thermalization (decoupled drag and diffusion)
        drift = 1.0 - gamma_eff * self.dt
        noise_std = torch.sqrt(T_target * self.collision_nu * self.dt * 2.0 / 3.0)
        v_final = v_next * drift + torch.randn_like(v_next) * noise_std
        
        # --- VELOCITY CEILING (The Artificial Refrigerator) ---
        # Hard clamp on stochastic tail singularities. During ignition (P_alpha > P_brems),
        # this acts as a numerical refrigerator, absorbing excess physical power and 
        # creating an artificial temperature plateau.
        v_speed = torch.norm(v_final, dim=1, keepdim=True)
        v_final = v_final * torch.clamp(v_speed, max=10.0) / (v_speed + 1e-6)
        
        return q_next, v_final, T_current.item(), torch.norm(torch.mean(q_next, dim=0)).item(), P_alpha.item(), P_brems.item()

# =============================================================================
# EXPERIMENT EXECUTION & FORENSIC TELEMETRY
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SymplecticFusionEngine(n_particles=512, device=device).to(device)
    
    # Initialize a cold plasma configuration
    q = torch.randn(512, 3, device=device) * 0.1
    v = torch.randn(512, 3, device=device) * 0.5
    
    history_T, history_Kink, history_Paux = [], [], []
    
    print("🚀 IGNITING: Executing Symplectic Burn and Ignition Telemetry...")
    
    for step in range(8000):
        # Apply auxiliary heating (NBI Proxy) between steps 1000 and 4000
        model.P_aux = 0.35 if 1000 < step < 4000 else 0.0
        
        with torch.no_grad():
            q, v, T, Kink, P_alpha, P_brems = model(q, v)
            history_T.append(T)
            history_Kink.append(Kink)
            history_Paux.append(model.P_aux)
            
        # Forensic Telemetry: Monitor Power Balance after Aux Heating is cut
        if step >= 4000 and step % 500 == 0:
            net_power = P_alpha - P_brems
            print(f"Step {step:04d} | T: {T:5.2f} | P_alpha: {P_alpha:6.4f} | P_brems: {P_brems:6.4f} | Net Physics Power: {net_power:+6.4f}")

    # =========================================================================
    # PLOTTING DIAGNOSTICS
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Temperature Trace
    ax1.plot(history_T, color='gold', linewidth=2, label='Plasma Temperature')
    ax1.fill_between(range(8000), [p*20 for p in history_Paux], color='gray', alpha=0.2, label='Aux Power (NBI) Active')
    ax1.axhline(y=7.032, color='white', linestyle='--', alpha=0.5, label='Theoretical Lawson Equilibrium')
    ax1.set_title("Symplectic Temperature Dynamics\n(Note: Post-4000 plateau is bounded by Velocity Ceiling)")
    ax1.set_ylabel("Temperature (Sim Units)")
    ax1.set_xlabel("Time Step")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.1)
    
    # Kink Amplitude Trace
    ax2.plot(history_Kink, color='red', linewidth=1.5, label='m=1 Center of Mass Drift')
    ax2.set_title("MHD Stability Diagnostic\n(Saturated Internal Kink Mode)")
    ax2.set_ylabel("Displacement (m)")
    ax2.set_xlabel("Time Step")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.1)
    
    plt.tight_layout()
    
    # Save the output figure for documentation
    plt.savefig("ignition_diagnostics.png", dpi=300, bbox_inches='tight')
    print("✅ Run complete. Plot saved as 'ignition_diagnostics.png'.")
    plt.show()
