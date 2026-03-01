"""
velocity_controller.py — Trapezoidal Motion Profile + Velocity PD Controller
=============================================================================
Reciprocating Actuator Simulation | Task 4

Architecture
------------
Two-layer velocity controller mirroring the moteus firmware velocity loop:

  Layer 1 — Trapezoidal Motion Profiler
  ──────────────────────────────────────
  Given current position θ and a target position θ_target, compute a
  velocity setpoint that follows a trapezoidal (ramp-up / cruise / ramp-down)
  profile constrained by:
      max_vel  [rad/s]  — cruise speed
      max_acc  [rad/s²] — acceleration / deceleration limit

  This models:
      moteus: set_position(velocity=...) with accel_limit inside firmware

  Layer 2 — PD Velocity Controller
  ──────────────────────────────────
  Given velocity setpoint from profiler, compute torque command:

      T = Kp·(ω_target - ω) + Kd·(-dω/dt)

  No integrator needed: position tracking (not velocity holding) is the goal.
  Saturation clips to maximum_torque.

Hard-Stop Model
---------------
  apply_hard_stop(theta) → clamps θ to [0, θ_stop]. Returns clamped θ.
  When motor reaches hard-stop, velocity is zeroed (inelastic collision).
  Simulates the mechanical end-stop use in torque homing.

Homing Finite State Machine
-----------------------------
  State HOMING_ACTIVE  → drives at slow negative velocity toward θ = 0.
  State HOMING_DONE    → triggers when |Iq| > I_threshold (hard-stop detected).
  home_position recorded as θ at that moment.

Author: Reciprocating Actuator Engineering Project
"""

import numpy as np


# ---------------------------------------------------------------------------
# Physical hard-stop positions — match the actuator stroke exactly
# ---------------------------------------------------------------------------
HARD_STOP_MIN  = 0.0   # [rad]  lower mechanical limit (home position)
HARD_STOP_MAX  = 2.5   # [rad]  upper mechanical limit (~143°, 0.5 rad margin above TARGET_A=2.0)
                        # CRITICAL: must be tight; a large value lets θ run away before clamping


def apply_hard_stop(theta: float, omega: float, restitution: float = 0.0):
    """
    Clamp shaft angle to [HARD_STOP_MIN, HARD_STOP_MAX].

    Returns
    -------
    (theta_clamped, omega_clamped, at_stop : bool)
    """
    at_stop = False
    if theta <= HARD_STOP_MIN:
        theta   = HARD_STOP_MIN
        if omega < 0:
            omega = -restitution * omega   # inelastic by default
        at_stop = True
    elif theta >= HARD_STOP_MAX:
        theta   = HARD_STOP_MAX
        if omega > 0:
            omega = -restitution * omega
        at_stop = True
    return theta, omega, at_stop


# ---------------------------------------------------------------------------
# Trapezoidal motion profiler
# ---------------------------------------------------------------------------
def trapezoidal_profile(
    theta:   float,
    target:  float,
    max_vel: float = 5.0,    # [rad/s]   cruise speed
    max_acc: float = 20.0,   # [rad/s²]  accel / decel limit
    dt:      float = 0.001,
) -> float:
    """
    Compute velocity command using a trapezoidal motion profile.

    The velocity is shaped so it:
      · Ramps up at max_acc toward max_vel when far from target
      · Ramps down at max_acc when approaching target (braking zone)
      · Clips to max_vel in cruise phase

    Parameters
    ----------
    theta   : float — Current shaft angle [rad]
    target  : float — Desired shaft angle [rad]
    max_vel : float — Maximum cruise velocity [rad/s]
    max_acc : float — Maximum acceleration [rad/s²]
    dt      : float — Time step [s]

    Returns
    -------
    vel_cmd : float — Target velocity [rad/s]
    """
    error = target - theta

    # Braking distance at current max_vel: s_brake = v² / (2·a)
    s_brake = (max_vel ** 2) / (2.0 * max_acc)

    if abs(error) < 1e-4:
        return 0.0                             # At target

    direction = np.sign(error)

    if abs(error) <= s_brake:
        # Deceleration zone: v = sqrt(2·a·|error|)
        vel_cmd = direction * np.sqrt(2.0 * max_acc * abs(error))
    else:
        # Cruise zone
        vel_cmd = direction * max_vel

    return float(np.clip(vel_cmd, -max_vel, max_vel))


# ---------------------------------------------------------------------------
# PD Velocity Controller
# ---------------------------------------------------------------------------
class VelocityPD:
    """
    Proportional-Derivative velocity controller.

    Maps velocity error → torque command, matching the moteus velocity loop.

    Parameters
    ----------
    Kp     : float — Proportional gain  [N·m·s/rad]
    Kd     : float — Derivative gain    [N·m·s²/rad]
    T_max  : float — Output torque saturation limit [N·m]
    """

    def __init__(
        self,
        Kp:    float = 0.3,
        Kd:    float = 0.005,
        T_max: float = 2.0,
    ):
        self.Kp    = Kp
        self.Kd    = Kd
        self.T_max = T_max
        self._prev_omega: float = 0.0

        # Diagnostics
        self.P_term: float = 0.0
        self.D_term: float = 0.0

    def compute(self, omega_target: float, omega_actual: float, dt: float) -> float:
        """
        Compute torque command.

        Parameters
        ----------
        omega_target : float — Desired angular velocity [rad/s]
        omega_actual : float — Measured angular velocity [rad/s]
        dt           : float — Time step [s]

        Returns
        -------
        torque : float — Commanded torque [N·m], saturated to ±T_max
        """
        vel_error = omega_target - omega_actual

        # Derivative of velocity = angular acceleration
        d_omega = (omega_actual - self._prev_omega) / dt

        self.P_term = self.Kp * vel_error
        self.D_term = -self.Kd * d_omega          # Derivative-on-measurement (no kick)

        output = self.P_term + self.D_term
        self._prev_omega = omega_actual

        return float(np.clip(output, -self.T_max, self.T_max))

    def reset(self) -> None:
        """
        Reset the controller state.
        """
        self._prev_omega = 0.0
        self.P_term = 0.0
        self.D_term = 0.0

    def __repr__(self) -> str:
        """
        Return a string representation of the controller.
        """
        return f"VelocityPD(Kp={self.Kp}, Kd={self.Kd}, T_max=±{self.T_max})"
