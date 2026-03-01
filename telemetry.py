"""
telemetry.py — Telemetry Logger & CSV Writer
============================================
Reciprocating Actuator Simulation | Task 4

Matches the moteus telemetry output format:
    target, actual, voltage, temp, iq

Each row corresponds to one control cycle (1 ms step in simulation).

Usage
-----
    logger = TelemetryLogger(path="log.csv")
    logger.log(t, target, actual, voltage, temp, iq)
    ...
    logger.close()
    df = logger.to_dataframe()   # Returns numpy structured array

Author: Reciprocating Actuator Engineering Project
"""

import os
import csv
import numpy as np


class TelemetryLogger:
    """
    Real-time row-by-row CSV logger and in-memory buffer.

    Parameters
    ----------
    path : str — Output CSV file path (relative or absolute).
    """

    COLUMNS = ["t", "target", "actual", "velocity", "voltage", "temp", "iq",
               "torque", "watchdog_fault", "homing_phase"]

    def __init__(self, path: str = "log.csv"):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        self._file   = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)

        # In-memory buffer for fast numpy processing after sim ends
        self._buffer: list[list] = []

    # -----------------------------------------------------------------------

    def log(
        self,
        t:              float,
        target:         float,
        actual:         float,
        velocity:       float,
        voltage:        float,
        temp:           float,
        iq:             float,
        torque:         float,
        watchdog_fault: int   = 0,
        homing_phase:   int   = 0,
    ) -> None:
        """
        Write one telemetry row.

        Parameters
        ----------
        t              : float — Simulation time [s]
        target         : float — Commanded / reference position [rad]
        actual         : float — Motor actual position [rad]
        velocity       : float — Actual angular velocity [rad/s]
        voltage        : float — Bus voltage [V]
        temp           : float — Winding temperature [°C]
        iq             : float — Phase current [A]
        torque         : float — Applied torque [N·m]
        watchdog_fault : int   — 1 if watchdog active, else 0
        homing_phase   : int   — 1 during homing, 0 during reciprocation
        """
        row = [
            round(t, 6),
            round(target, 6),
            round(actual, 6),
            round(velocity, 6),
            round(voltage, 4),
            round(temp, 4),
            round(iq, 6),
            round(torque, 6),
            int(watchdog_fault),
            int(homing_phase),
        ]
        self._writer.writerow(row)
        self._buffer.append(row)

    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the CSV file."""
        self._file.flush()
        self._file.close()

    # -----------------------------------------------------------------------

    def to_numpy(self) -> dict:
        """
        Convert in-memory buffer to a dict of numpy arrays (one per column).

        Returns
        -------
        data : dict[str, np.ndarray]
        """
        if not self._buffer:
            return {col: np.array([]) for col in self.COLUMNS}

        arr = np.array(self._buffer, dtype=float)
        return {col: arr[:, i] for i, col in enumerate(self.COLUMNS)}
