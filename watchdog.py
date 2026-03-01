"""
watchdog.py — CAN/SPI Communication Watchdog & Current Limit Guard
===================================================================
Reciprocating Actuator Simulation | Task 4

Purpose
-------
Simulates the moteus controller watchdog behavior:

    "If no valid command is received within 100 ms, the controller
     enters a fault state and stops the motor."

    — moteus documentation: watchdog_timeout

Two guards are combined here:

1. CommandWatchdog
   ───────────────
   Tracks wall-clock (simulation) time since last valid command update.
   If time_since_last_command > TIMEOUT_S (100 ms), fires a fault.

   In real moteus code this maps to:
       controller.set_stop()  →  called automatically on watchdog timeout

2. CurrentLimitGuard
   ──────────────────
   Monitors Iq (phase current) every step.
   If |Iq| > I_THRESHOLD (1.5 A), triggers a fault (hard-stop detected).

   This is used for torque-based homing:
       During homing, motor spins slowly to θ = 0 (hard-stop).
       On contact, back-torque causes Iq to spike above threshold.
       Guard detects the spike → homing_complete = True.

Fault Handling
--------------
Both faults set fault_active = True and return stop_motor = True to the
sim loop. The sim loop then zeros the torque command, mirroring
await controller.set_stop() in async moteus Python client code.

Author: Reciprocating Actuator Engineering Project
"""

import time


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WATCHDOG_TIMEOUT_S  = 0.100   # 100 ms — moteus default watchdog timeout
CURRENT_THRESHOLD_A = 1.5     # [A] — homing hard-stop detection threshold


# ---------------------------------------------------------------------------
# Command Watchdog
# ---------------------------------------------------------------------------
class CommandWatchdog:
    """
    Simulates the moteus 100 ms command watchdog.

    Usage
    -----
    Call `pet()` every time a new command is sent to the controller.
    Call `check(sim_time)` in the inner loop to detect timeouts.

    In real hardware:
        await controller.set_position(...)  →  pet()
        await asyncio.sleep(0.01)           →  next check() call
    """

    def __init__(self, timeout_s: float = WATCHDOG_TIMEOUT_S):
        self.timeout_s        = timeout_s
        self._last_command_t  = 0.0   # Simulation time [s] of last command
        self.fault_active     = False
        self.fault_count      = 0
        self.time_since_last  = 0.0   # Exposed for logging

    def pet(self, sim_time: float) -> None:
        """
        Record that a command was sent at simulation time `sim_time`.
        Resets the watchdog timer.
        """
        self._last_command_t = sim_time
        self.fault_active    = False

    def check(self, sim_time: float) -> bool:
        """
        Check whether the watchdog has expired.

        Parameters
        ----------
        sim_time : float — Current simulation time [s]

        Returns
        -------
        fault : bool — True if watchdog timeout has fired
        """
        self.time_since_last = sim_time - self._last_command_t

        if self.time_since_last > self.timeout_s:
            if not self.fault_active:
                self.fault_count += 1
            self.fault_active = True

        return self.fault_active

    def reset(self, sim_time: float) -> None:
        """Clear fault and restart timer."""
        self._last_command_t = sim_time
        self.fault_active    = False

    def __repr__(self) -> str:
        """
        Return a string representation of the watchdog.
        """
        return (
            f"CommandWatchdog(timeout={self.timeout_s*1000:.0f}ms, "
            f"fault={self.fault_active}, faults={self.fault_count})"
        )


# ---------------------------------------------------------------------------
# Current (Iq) Limit Guard
# ---------------------------------------------------------------------------
class CurrentLimitGuard:
    """
    Monitors motor phase current and triggers a fault when |Iq| > threshold.

    Primarily used during torque-based homing:
        When the shaft hits the mechanical hard-stop (θ = 0), the
        back-torque causes a sharp Iq spike that this guard detects.

    Parameters
    ----------
    threshold_a : float — Current fault threshold [A]
    """

    def __init__(self, threshold_a: float = CURRENT_THRESHOLD_A):
        self.threshold_a  = threshold_a
        self.fault_active = False
        self.peak_iq      = 0.0

    def check(self, iq: float) -> bool:
        """
        Check phase current against threshold.

        Parameters
        ----------
        iq : float — Phase current [A]

        Returns
        -------
        fault : bool — True if |Iq| exceeds threshold
        """
        if abs(iq) > self.peak_iq:
            self.peak_iq = abs(iq)

        if abs(iq) > self.threshold_a:
            self.fault_active = True

        return self.fault_active

    def reset(self) -> None:
        """Clear fault (e.g., after homing completes)."""
        self.fault_active = False
        self.peak_iq      = 0.0

    def __repr__(self) -> str:
        """
        Return a string representation of the guard.
        """
        return (
            f"CurrentLimitGuard(threshold={self.threshold_a:.2f}A, "
            f"peak_Iq={self.peak_iq:.3f}A, fault={self.fault_active})"
        )
