"""
motor.py — High-Speed BLDC Motor Dynamics Model
================================================
Reciprocating Actuator Simulation | Task 4

Physical Model
--------------
Newton-Euler rotational equation of motion:

    J · dω/dt = T_motor - B·ω - T_friction·sign(ω) - T_load

State Variables
---------------
    θ (position [rad])
    ω (angular velocity [rad/s])

Differences from Task 3
------------------------
- Higher max torque (T_max = 2.0 N·m) matching moteus 2.0 A current limit
- Higher inertia + lower viscous damping to allow fast reciprocation
- Full back-EMF model: V_bemf = Ke·ω, voltage & temperature tracked
- Torque constant Kt used to compute phase current Iq from commanded torque
- Voltage sag model: V_bus drops under high load (simulates real battery/PSU)

Hardware Analogy (moteus r4.11)
--------------------------------
    set_position(position=nan, velocity=v, maximum_torque=2.0)
    → drives velocity loop; maximum_torque maps to current limit
    → current Iq = T_commanded / Kt

Author: Reciprocating Actuator Engineering Project
"""

import numpy as np


# ---------------------------------------------------------------------------
# Motor electrical constants
# ---------------------------------------------------------------------------
Kt   = 1.0     # Torque constant [N·m / A]   τ = Kt · Iq  (realistic: 2 N·m → 2 A)
Ke   = 1.0     # Back-EMF constant [V·s/rad]  V_bemf = Ke · ω  (= Kt for SI units)
R_ph = 0.5     # Phase resistance [Ω]
V_BUS_NOMINAL = 24.0   # Nominal bus voltage [V]
V_SAG_COEFF   = 0.10   # Voltage sag per amp [V/A]  V_bus = V_nom - k_sag·|Iq|

# Velocity saturation (prevents position runaway from unbounded integration)
# Derived from: with V_bus=24V, Ke=1.0, no-load speed ≈ V_bus/Ke = 24 rad/s
OMEGA_MAX = 15.0   # [rad/s]  hard velocity clamp (>= MAX_VEL of 5 rad/s cruise)


class BLDCMotorHS:
    """
    Discrete-time high-speed BLDC motor model for reciprocating actuator.

    Parameters
    ----------
    J          : float — Moment of inertia [kg·m²]
    B          : float — Viscous damping   [N·m·s/rad]
    T_friction : float — Coulomb friction  [N·m]
    T_max      : float — Max torque (mirrors moteus maximum_torque) [N·m]
    theta0     : float — Initial shaft angle [rad]
    omega0     : float — Initial angular velocity [rad/s]
    temp0      : float — Initial winding temperature [°C]
    thermal_R  : float — Thermal resistance [°C/W]
    thermal_C  : float — Thermal capacitance [J/°C]
    """

    def __init__(
        self,
        J:          float = 2e-4,          # Realistic actuator inertia; α_max = T_max/J = 10,000 rad/s²
        B:          float = 5e-3,          # Viscous damping → terminal velocity ≈ T_max/B ≈ 400 rad/s (back-EMF limits further)
        T_friction: float = 0.01,
        T_max:      float = 2.0,           # moteus maximum_torque
        theta0:     float = 0.0,
        omega0:     float = 0.0,
        temp0:      float = 25.0,
        thermal_R:  float = 2.5,
        thermal_C:  float = 50.0,
    ):
        self.J          = J
        self.B          = B
        self.T_friction = T_friction
        self.T_max      = T_max

        # State
        self.theta = theta0    # [rad]
        self.omega = omega0    # [rad/s]
        self.temp  = temp0     # [°C] winding temperature

        # Thermal model
        self._thermal_R = thermal_R   # [°C/W]
        self._thermal_C = thermal_C   # [J/°C]

        # Electrical state (updated after step)
        self.iq      = 0.0   # [A]   phase current
        self.voltage = V_BUS_NOMINAL  # [V]  bus voltage (sags under load)

    # -----------------------------------------------------------------------
    # Main integration step
    # -----------------------------------------------------------------------
    def step(self, T_input: float, dt: float = 0.001) -> tuple:
        """
        Advance motor state by one time step.

        Parameters
        ----------
        T_input : float — Commanded torque [N·m]
        dt      : float — Time step [s]

        Returns
        -------
        (theta, omega) : tuple[float, float]
        """
        # 1. Saturate torque (maximum_torque limiting)
        T_motor = np.clip(T_input, -self.T_max, self.T_max)

        # 2. Compute drive current Iq from clamped torque
        self.iq = T_motor / Kt

        # 3. Bus voltage sag under load
        self.voltage = V_BUS_NOMINAL - V_SAG_COEFF * abs(self.iq)

        # 4. Coulomb friction (smooth tanh approximation near zero speed)
        T_fric = self.T_friction * np.tanh(self.omega / 0.05)

        # 5. Back-EMF torque loss contribution
        #    V_available for torque = V_bus - V_bemf - I·R
        #    Here we fold this into an effective damping already captured by B,
        #    but we also compute V_bemf for telemetry accuracy.
        V_bemf = Ke * abs(self.omega)
        # Effective torque reduced if back-EMF approaches supply voltage
        back_emf_factor = np.clip(1.0 - V_bemf / max(self.voltage, 0.1), 0.0, 1.0)
        T_net_drive = T_motor * back_emf_factor

        # 6. Newton-Euler
        T_net = T_net_drive - self.B * self.omega - T_fric
        alpha = T_net / self.J

        # 7. Forward Euler integration
        self.omega += alpha * dt

        # 8. HARD VELOCITY CLAMP — prevents unbounded ω integration
        #    This mirrors the moteus firmware's maximum_velocity parameter.
        #    Without this, small J + large T produces ω → ∞ → θ → ∞.
        self.omega = np.clip(self.omega, -OMEGA_MAX, OMEGA_MAX)

        self.theta += self.omega * dt

        # 9. Thermal model  (P_loss = Iq² · R_ph)
        P_loss = (self.iq ** 2) * R_ph
        dT_dt  = (P_loss * self._thermal_R - (self.temp - 25.0)) / self._thermal_C
        self.temp += dT_dt * dt
        # Clamp temperature to realistic range
        self.temp = min(self.temp, 150.0)

        return self.theta, self.omega

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------
    def reset(self, theta: float = 0.0, omega: float = 0.0) -> None:
        """Reset state (used after homing completes)."""
        self.theta = theta
        self.omega = omega
        self.iq    = 0.0

    def kinetic_energy(self) -> float:
        """Rotational kinetic energy ½Jω² [J]."""
        return 0.5 * self.J * self.omega ** 2

    def compute_current(self, torque: float) -> float:
        """Derive phase current from torque request:  Iq = τ / Kt."""
        return torque / Kt

    def __repr__(self) -> str:
        """Return a string representation of the motor."""
        return (
            f"BLDCMotorHS(θ={np.degrees(self.theta):.3f}°, "
            f"ω={self.omega:.2f} rad/s, "
            f"Iq={self.iq:.3f} A, "
            f"T={self.temp:.1f}°C, "
            f"V={self.voltage:.2f} V)"
        )
