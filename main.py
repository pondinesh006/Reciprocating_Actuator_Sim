"""
main.py — High-Speed Reciprocating Actuator Simulation Orchestrator
====================================================================
Reciprocating Actuator Simulation | Task 4 — moteus Controller Emulation

Three Demonstrated Behaviours
------------------------------
  1. TORQUE-BASED HOMING (visible ~1 s journey)
     Motor starts at θ = 2.0 rad.  Driven at −2.0 rad/s toward hard-stop.
     On contact at θ = 0, Iq spikes above 1.5 A → homing complete.
     Plot shows: negative velocity ramp, Iq burst, "Homing complete" marker.

  2. RECIPROCATING MOTION (10 s, 36+ strokes)
     Trapezoidal velocity profile between 0.0 ↔ 2.0 rad at 5 rad/s.
     Every step pets the watchdog → 0 natural faults.

  3. WATCHDOG FAULT DEMONSTRATION (at t = 7 s into reciprocation)
     Commands deliberately paused for 250 ms → watchdog fires.
     Torque is zeroed, motor coasts to a stop, then resumes.
     Plot shows: torque → 0, velocity decay, "[WD] Watchdog FAULT" marker.

Output Files
------------
  plots/reciprocating_actuator_results.png  — 5-panel combined timeline
  log.csv                                   — full telemetry (homing + recip)

Run
---
  python main.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from motor               import BLDCMotorHS
from velocity_controller import (VelocityPD, trapezoidal_profile,
                                  apply_hard_stop)
from watchdog            import CommandWatchdog, CurrentLimitGuard
from telemetry           import TelemetryLogger


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

DT             = 0.001    # [s]  1 ms timestep (1 kHz)
T_HOMING_MAX   = 3.0      # [s]  max homing duration
T_RECIPROCAL   = 10.0     # [s]  reciprocating phase duration

MOTOR_START_THETA = 2.0   # [rad] motor starts HERE — gives ~1 s visible homing journey
HOMING_VEL        = -2.0  # [rad/s] fast homing → Iq spike clearly exceeds 1.5 A threshold
                           #   when stalled at θ=0: vel_error=2.0, T = Kp×2.0 = 3.0 → clip 2.0 N·m
                           #   Iq = 2.0 / Kt = 2.0 A > 1.5 A ✓

TARGET_A   = 2.0           # [rad]  stroke end A
TARGET_B   = 0.0           # [rad]  stroke end B
SETTLE_TOL = 0.02          # [rad]  arrival tolerance → direction switch
MAX_VEL    = 5.0           # [rad/s]
MAX_ACC    = 20.0          # [rad/s²]

# Watchdog demo: deliberately pause commands at this time (relative to recip phase)
WATCHDOG_DEMO_REL_T = 7.0       # [s]  silence starts at t_recip + 7 s
WATCHDOG_GAP_S      = 0.25      # [s]  250 ms gap (> 100 ms timeout → guaranteed fault)

STEPS_HOMING = int(T_HOMING_MAX / DT)
STEPS_RECIP  = int(T_RECIPROCAL / DT)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
LOG_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.csv")

# Shared log column template (same for homing and reciprocating)
LOG_KEYS = ["t", "target", "actual", "velocity", "torque",
            "iq", "voltage", "temp", "watchdog", "homing_phase"]


# =============================================================================
# HELPER — empty numpy log pre-allocation
# =============================================================================
def _alloc_log(n: int) -> dict:
    """
    Pre-allocate numpy array for log of length `n`.
    """
    return {k: np.zeros(n) for k in LOG_KEYS}


def _concat_logs(a: dict, b: dict) -> dict:
    """
    Concatenate two logs of different lengths.
    """
    return {k: np.concatenate([a[k], b[k]]) for k in LOG_KEYS}


# =============================================================================
# PHASE 1 — TORQUE-BASED HOMING (visibly ~1 s)
# =============================================================================
def run_homing(motor, vel_pd, current_guard, watchdog, logger):
    """
    Drive motor from MOTOR_START_THETA toward hard-stop at θ = 0.

    Physics of current spike
    ------------------------
    While moving freely: ω ≈ HOMING_VEL, vel_error ≈ 0, Iq ≈ 0.
    On hard-stop contact (θ = 0): ω zeroed each step.
      Next step: vel_error = |HOMING_VEL| = 2.0 rad/s
          T = Kp × 2.0 = 3.0 → clipped to T_max = 2.0 N·m
          Iq = T / Kt = 2.0 A  >  1.5 A threshold  ✓

    Returns
    -------
    (log, home_position, t_home_end)
    """
    print(f"  [HOMING] Motor at θ = {np.degrees(motor.theta):.1f}°  "
          f"→ driving to hard-stop at θ = 0° ...")

    log = _alloc_log(STEPS_HOMING)
    home_position = motor.theta
    steps_used = STEPS_HOMING

    for k in range(STEPS_HOMING):
        t = k * DT
        watchdog.pet(t)

        vel_cmd = HOMING_VEL
        torque   = vel_pd.compute(vel_cmd, motor.omega, DT)
        motor.step(torque, DT)

        # Physical hard-stop at θ = 0
        theta_c, omega_c, at_stop = apply_hard_stop(motor.theta, motor.omega)
        motor.theta = theta_c
        motor.omega = omega_c

        iq_fault = current_guard.check(motor.iq)

        log["t"][k]           = t
        log["target"][k]      = TARGET_B          # homing target = 0
        log["actual"][k]      = motor.theta
        log["velocity"][k]    = motor.omega
        log["torque"][k]      = torque
        log["iq"][k]          = motor.iq
        log["voltage"][k]     = motor.voltage
        log["temp"][k]        = motor.temp
        log["watchdog"][k]    = 0
        log["homing_phase"][k]= 1

        logger.log(t, TARGET_B, motor.theta, motor.omega,
                   motor.voltage, motor.temp, motor.iq, torque,
                   watchdog_fault=0, homing_phase=1)

        if iq_fault:
            home_position = motor.theta
            steps_used    = k + 1
            print(f"  [HOMING] ⚡ Hard-stop DETECTED at θ = {np.degrees(home_position):.2f}°"
                  f"   |Iq| = {abs(motor.iq):.3f} A   (t = {t:.3f}s)")
            break

    if not current_guard.fault_active:
        print("  [HOMING] ⚠ Warning: timeout — motor may not have reached hard-stop.")
        home_position = motor.theta

    # Trim log to actual steps used
    trimmed = {k: log[k][:steps_used] for k in LOG_KEYS}
    t_home_end = trimmed["t"][-1]
    print(f"  [HOMING] Homing complete at t = {t_home_end:.3f}s  "
          f"→ home_position = {np.degrees(home_position):.3f}°\n")
    return trimmed, home_position, t_home_end


# =============================================================================
# PHASE 2 — RECIPROCATING MOTION with WATCHDOG DEMO
# =============================================================================
def run_reciprocating(motor, vel_pd, current_guard, watchdog, logger, t_offset):
    """
    Reciprocate 0.0 ↔ 2.0 rad for T_RECIPROCAL seconds.

    At WATCHDOG_DEMO_REL_T seconds into this phase:
        Commands are deliberately paused for WATCHDOG_GAP_S (250 ms).
        The watchdog fires → torque set to 0 → motor coasts to halt.
        After the gap, commands resume and motion recovers.

    Returns
    -------
    (log, t_wd_start, t_wd_end)  — watchdog event absolute times
    """
    print(f"  [RECIP]  Starting reciprocating motion for {T_RECIPROCAL}s ...")
    print(f"  [RECIP]  Watchdog demo: commands paused at "
          f"t_recip = {WATCHDOG_DEMO_REL_T}s "
          f"for {WATCHDOG_GAP_S*1000:.0f} ms ...\n")

    log = _alloc_log(STEPS_RECIP)

    direction            = 1
    target               = TARGET_A
    dwell_steps_left     = 0

    # Watchdog demo tracking
    wd_demo_start_abs  = t_offset + WATCHDOG_DEMO_REL_T
    wd_demo_end_abs    = wd_demo_start_abs + WATCHDOG_GAP_S
    t_wd_start = wd_demo_start_abs
    t_wd_end   = wd_demo_end_abs
    wd_demo_fired      = False

    current_guard.reset()
    vel_pd.reset()

    for k in range(STEPS_RECIP):
        t = t_offset + k * DT
        in_wd_gap = wd_demo_start_abs <= t < wd_demo_end_abs

        # --- Watchdog pet logic ---
        if not in_wd_gap:
            watchdog.pet(t)           # Normal operation: always pet

        wd_fault = watchdog.check(t)

        if wd_fault and not wd_demo_fired:
            wd_demo_fired = True
            print(f"  [WD]  [WD] Watchdog FAULT at t = {t:.3f}s  "
                  f"(gap = {watchdog.time_since_last*1000:.1f} ms > 100 ms)")

        # --- Control ---
        if wd_fault:
            # Watchdog active → zero torque (motor.stop equivalent)
            torque  = 0.0
            vel_cmd = 0.0
        elif dwell_steps_left > 0:
            dwell_steps_left -= 1
            torque  = 0.0
            vel_cmd = 0.0
        else:
            if abs(motor.theta - target) < SETTLE_TOL:
                direction        *= -1
                target            = TARGET_A if direction == 1 else TARGET_B
                dwell_steps_left  = int(0.1 / DT)
                torque  = 0.0
                vel_cmd = 0.0
            else:
                vel_cmd = trapezoidal_profile(motor.theta, target, MAX_VEL, MAX_ACC, DT)
                torque  = vel_pd.compute(vel_cmd, motor.omega, DT)

        # --- Motor dynamics ---
        motor.step(torque, DT)
        theta_c, omega_c, _ = apply_hard_stop(motor.theta, motor.omega)
        motor.theta = theta_c
        motor.omega = omega_c

        log["t"][k]           = t
        log["target"][k]      = target
        log["actual"][k]      = motor.theta
        log["velocity"][k]    = motor.omega
        log["torque"][k]      = torque
        log["iq"][k]          = motor.iq
        log["voltage"][k]     = motor.voltage
        log["temp"][k]        = motor.temp
        log["watchdog"][k]    = float(wd_fault)
        log["homing_phase"][k]= 0

        logger.log(t, target, motor.theta, motor.omega,
                   motor.voltage, motor.temp, motor.iq, torque,
                   watchdog_fault=int(wd_fault), homing_phase=0)

    print(f"  [RECIP]  Done.  Watchdog faults triggered: {watchdog.fault_count}")
    return log, t_wd_start, t_wd_end


# =============================================================================
# VALIDATION
# =============================================================================
def run_validation(recip_log: dict) -> dict:
    """Compute performance metrics on the reciprocating phase only."""

    target   = recip_log["target"]
    actual   = recip_log["actual"]
    velocity = recip_log["velocity"]
    iq       = recip_log["iq"]
    voltage  = recip_log["voltage"]
    temp     = recip_log["temp"]
    wd       = recip_log["watchdog"]

    results = {}

    # 1. Endpoint tracking error (settled periods only)
    at_rest      = np.abs(velocity) < 0.3
    at_a         = (np.abs(actual - TARGET_A) < 0.15) & at_rest
    at_b         = (np.abs(actual - TARGET_B) < 0.15) & at_rest
    settled_mask = at_a | at_b
    if np.any(settled_mask):
        ep_ref = np.where(at_a[settled_mask], TARGET_A, TARGET_B)
        mae    = float(np.mean(np.abs(actual[settled_mask] - ep_ref)))
    else:
        mae = float(np.mean(np.abs(actual - target)))
    results["mean_tracking_error_rad"] = (round(mae, 5), mae < 0.05)

    # 2. Peak Iq
    peak_iq = float(np.max(np.abs(iq)))
    results["peak_iq_A"] = (round(peak_iq, 4), peak_iq < 2.01)

    # 3. Minimum bus voltage
    min_v = float(np.min(voltage))
    results["min_bus_voltage_V"] = (round(min_v, 3), min_v > 22.0)

    # 4. Max winding temperature
    max_temp = float(np.max(temp))
    results["max_winding_temp_C"] = (round(max_temp, 2), max_temp < 100.0)

    # 5. Watchdog faults triggered (count rising edges)
    wd_events = int(np.sum(np.diff((wd > 0).astype(int)) > 0))
    results["watchdog_faults_triggered"] = (wd_events, wd_events >= 1)  # ≥1 demonstrates WD works

    # 6. Stroke reversals
    rev = int(np.sum(np.abs(np.diff(target)) > 0.5))
    results["stroke_reversals"] = (rev, rev >= 10)

    # --- Print ---
    print("\n" + "=" * 62)
    print("  RECIPROCATING ACTUATOR SIMULATION — VALIDATION REPORT")
    print("=" * 62)
    labels = {
        "mean_tracking_error_rad":    "Mean endpoint tracking error  [rad]",
        "peak_iq_A":                  "Peak phase current Iq         [A]",
        "min_bus_voltage_V":          "Minimum bus voltage           [V]",
        "max_winding_temp_C":         "Max winding temperature       [°C]",
        "watchdog_faults_triggered":  "Watchdog faults triggered     [count]",
        "stroke_reversals":           "Stroke reversals              [count]",
    }
    all_ok = True
    for key, (val, ok) in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        if not ok:
            all_ok = False
        print(f"  {status}  {labels[key]:<38} = {val}")
    print("-" * 62)
    print(f"  Overall: {'ALL TESTS PASSED ✓' if all_ok else 'SOME TESTS FAILED ✗'}")
    print("=" * 62 + "\n")
    return results


# =============================================================================
# 5-PANEL COMBINED PLOT
# =============================================================================
def create_plots(combined_log: dict, recip_log: dict,
                 results: dict,
                 t_home_end: float,
                 t_wd_start: float, t_wd_end: float) -> str:
    """
    5-panel engineering figure covering the FULL combined timeline.

    Panels
    ------
    1. Position tracking (target vs actual)
    2. Angular velocity ω
    3. Phase current Iq  (homing spike + accel bursts visible)
    4. Applied torque    (watchdog fault zeroing clearly visible)
    5. Bus voltage + winding temperature (dual axis)

    Markers
    -------
    • Vertical dashed line at t_home_end  →  "Homing complete [OK]"
    • Vertical dashed line at t_wd_start  →  "[WD] Watchdog FAULT"
    • Shaded region [t_wd_start, t_wd_end] — fault duration
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    t       = combined_log["t"]
    target  = combined_log["target"]
    actual  = combined_log["actual"]
    vel     = combined_log["velocity"]
    torque  = combined_log["torque"]
    iq      = combined_log["iq"]
    voltage = combined_log["voltage"]
    temp    = combined_log["temp"]
    hp      = combined_log["homing_phase"]   # 1 = homing, 0 = recip

    C = {
        "ref":    "#F72585",
        "act":    "#4CC9F0",
        "vel":    "#FF9F1C",
        "iq":     "#7B2FBE",
        "iq_th":  "#E63946",
        "torque": "#4361EE",
        "volt":   "#4CAF50",
        "temp":   "#F4A261",
        "warn":   "#FFB703",
        "fault":  "#E63946",
        "home":   "#06D6A0",
        "grid":   "#21262D",
        "ax":     "#161B22",
        "fg":     "#C9D1D9",
        "title":  "#E6EDF3",
        "spine":  "#30363D",
    }

    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(18, 24), facecolor="#0D1117")
    gs  = gridspec.GridSpec(5, 1, hspace=0.60, figure=fig,
                            top=0.95, bottom=0.04, left=0.09, right=0.96)

    t_end = t[-1]

    # ---------------------------------------------------------------------------
    # Helper: style axes
    # ---------------------------------------------------------------------------
    def style_ax(ax, title, ylabel, xlabel=None):
        """
        Style a matplotlib axis.
        """
        ax.set_facecolor(C["ax"])
        ax.set_title(title, color=C["title"], fontsize=11, fontweight="bold", pad=6)
        ax.set_ylabel(ylabel, color=C["fg"], fontsize=9)
        if xlabel:
            ax.set_xlabel(xlabel, color=C["fg"], fontsize=9)
        ax.tick_params(colors=C["fg"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(C["spine"])
        ax.legend(fontsize=8, framealpha=0.3, facecolor=C["ax"],
                  labelcolor=C["fg"], loc="best")
        ax.grid(True, color=C["grid"], linewidth=0.6)
        ax.set_xlim(t[0], t_end)

    # ---------------------------------------------------------------------------
    # Helper: draw phase markers on every axis
    # ---------------------------------------------------------------------------
    def add_phase_markers(ax, ymin, ymax):
        """
        Draw phase markers on every axis.
        """
        # Homing region (light teal)
        ax.axvspan(t[0], t_home_end, alpha=0.10, color=C["home"], zorder=0)
        # Watchdog fault region (light red)
        ax.axvspan(t_wd_start, t_wd_end, alpha=0.18, color=C["fault"], zorder=0)

        # "Homing complete" line
        ax.axvline(t_home_end, color=C["home"], lw=1.6, ls="--", zorder=5)
        ax.text(t_home_end + 0.05, ymax * 0.93,
                "Homing\ncomplete ✓", color=C["home"], fontsize=7.5,
                fontweight="bold", va="top")

        # "Watchdog FAULT" line
        ax.axvline(t_wd_start, color=C["fault"], lw=1.8, ls="--", zorder=5)
        ax.text(t_wd_start + 0.05, ymax * 0.93,
                "[WD] Watchdog\nFAULT", color=C["fault"], fontsize=7.5,
                fontweight="bold", va="top")

        # "Watchdog resume" line
        ax.axvline(t_wd_end, color=C["warn"], lw=1.2, ls=":", zorder=5)
        ax.text(t_wd_end + 0.05, ymax * 0.80,
                "Resume", color=C["warn"], fontsize=7, va="top")

    # ---- Panel 1: Position Tracking ----------------------------------------
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t[hp == 1], np.degrees(actual[hp == 1]),
             color=C["act"], lw=1.8, label="Actual θ (HOMING)", alpha=0.9)
    ax1.plot(t[hp == 0], np.degrees(target[hp == 0]),
             color=C["ref"], lw=1.8, ls="--", label="Target θ", zorder=4)
    ax1.plot(t[hp == 0], np.degrees(actual[hp == 0]),
             color=C["act"], lw=1.4, label="Actual θ (RECIP)", zorder=3)
    ax1.axhline(0,            color=C["home"], lw=0.8, ls=":", alpha=0.6)
    ax1.axhline(np.degrees(TARGET_A), color=C["ref"], lw=0.8, ls=":", alpha=0.4)
    add_phase_markers(ax1,
                      ymin=np.degrees(actual.min()) - 5,
                      ymax=np.degrees(MOTOR_START_THETA) + 15)
    mae, _ = results["mean_tracking_error_rad"]
    ax1.annotate(f"Endpoint MAE = {np.degrees(mae):.3f}°",
                 xy=(0.60, 0.05), xycoords="axes fraction",
                 color="#F8961E", fontsize=8, style="italic")
    style_ax(ax1,
             "Panel 1 — Position Tracking  │  Homing (green) → Reciprocating → Watchdog fault (red)",
             "Angle [°]")

    # ---- Panel 2: Angular Velocity -----------------------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, vel, color=C["vel"], lw=1.2, label="ω [rad/s]")
    ax2.axhline(+MAX_VEL, color=C["iq_th"], lw=0.7, ls=":", alpha=0.5,
                label=f"±cruise limit {MAX_VEL} rad/s")
    ax2.axhline(-MAX_VEL, color=C["iq_th"], lw=0.7, ls=":", alpha=0.5)
    ax2.axhline(0, color=C["spine"], lw=0.8)
    add_phase_markers(ax2, -MAX_VEL - 1, MAX_VEL + 1)
    style_ax(ax2, "Panel 2 — Angular Velocity ω  │  Homing ramp → Reciprocation → WD coast-down",
             "ω [rad/s]")

    # ---- Panel 3: Phase Current Iq -----------------------------------------
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t, iq, color=C["iq"], lw=1.2, label="Iq [A]", zorder=3)
    ax3.axhline(+1.5, color=C["iq_th"], lw=1.0, ls="--",
                label="Homing threshold ±1.5 A", alpha=0.9)
    ax3.axhline(-1.5, color=C["iq_th"], lw=1.0, ls="--", alpha=0.9)
    ax3.axhline(+2.0, color="#FF4500", lw=0.7, ls=":", alpha=0.6,
                label="T_max limit ±2.0 A")
    ax3.axhline(-2.0, color="#FF4500", lw=0.7, ls=":", alpha=0.6)
    ax3.fill_between(t, iq, 0, where=(np.abs(iq) >= 1.5),
                     alpha=0.25, color=C["iq_th"], label="Threshold exceeded")
    pk, _ = results["peak_iq_A"]
    ax3.annotate(f"Peak |Iq| = {pk:.3f} A  ← homing spike",
                 xy=(0.01, 0.88), xycoords="axes fraction",
                 color="#F8961E", fontsize=8, style="italic")
    add_phase_markers(ax3, -2.2, 2.2)
    style_ax(ax3,
             "Panel 3 — Phase Current Iq  │  Homing spike > 1.5 A triggers stop",
             "Iq [A]")

    # ---- Panel 4: Torque (watchdog zeroing clearly visible) ----------------
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(t, torque, color=C["torque"], lw=1.2, label="Applied torque [N·m]")
    ax4.axhline(+2.0, color=C["iq_th"], lw=0.7, ls=":", alpha=0.5,
                label="±T_max = 2.0 N·m")
    ax4.axhline(-2.0, color=C["iq_th"], lw=0.7, ls=":", alpha=0.5)
    ax4.axhline(0, color=C["spine"], lw=0.8)
    # Annotate the watchdog zero-torque drop
    ax4.annotate("Torque → 0\n(WD fault)",
                 xy=(t_wd_start + 0.02, -0.2),
                 xycoords="data",
                 color=C["fault"], fontsize=8, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C["fault"], lw=1.2),
                 xytext=(t_wd_start + 0.6, -1.2))
    add_phase_markers(ax4, -2.3, 2.3)
    style_ax(ax4,
             "Panel 4 — Applied Torque  │  Watchdog fault zeroes torque (motor coast-down)",
             "Torque [N·m]")

    # ---- Panel 5: Voltage + Temperature (dual axis) ------------------------
    ax5 = fig.add_subplot(gs[4])
    ax5.plot(t, voltage, color=C["volt"], lw=1.4, label="V_bus [V]")
    ax5.axhline(24.0, color=C["spine"], lw=0.8, ls="--", alpha=0.5,
                label="Nominal 24 V")
    min_v, _ = results["min_bus_voltage_V"]
    ax5.annotate(f"Min V = {min_v:.2f} V",
                 xy=(0.02, 0.12), xycoords="axes fraction",
                 color="#F8961E", fontsize=8, style="italic")

    # Style ax5 and add markers BEFORE twinx (avoids matplotlib xlim/tick crash)
    add_phase_markers(ax5, voltage.min() - 0.3, voltage.max() + 0.3)
    style_ax(ax5,
             "Panel 5 \u2014 Bus Voltage & Winding Temperature",
             "V_bus [V]",
             xlabel="Time [s]")

    # Create twin axis AFTER style_ax to avoid renderer conflict
    ax5_t = ax5.twinx()
    ax5_t.set_facecolor("none")
    ax5_t.plot(t, temp, color=C["temp"], lw=1.3, ls="--", alpha=0.9,
               label="T_winding [\u00b0C]")
    ax5_t.set_ylabel("Temperature [\u00b0C]", color=C["temp"], fontsize=9)
    ax5_t.tick_params(colors=C["temp"], labelsize=8)
    ax5_t.spines["right"].set_edgecolor(C["temp"])
    max_t_val, _ = results["max_winding_temp_C"]
    ax5_t.annotate(f"Max T = {max_t_val:.1f}\u00b0C",
                   xy=(0.60, 0.88), xycoords="axes fraction",
                   color="#F8961E", fontsize=8, style="italic")
    # Merged legend
    ln_v, lb_v = ax5.get_legend_handles_labels()
    ln_t, lb_t = ax5_t.get_legend_handles_labels()
    try:
        ax5.get_legend().remove()
    except Exception:
        pass
    ax5.legend(ln_v + ln_t, lb_v + lb_t,
               fontsize=8, framealpha=0.3, facecolor=C["ax"],
               labelcolor=C["fg"], loc="lower right")

    # Global title
    fig.suptitle(
        "High-Speed Reciprocating Actuator Simulation — Task 4\n"
        "Torque Homing  |  Trapezoidal Profile  |  Watchdog Fault Demo  |  Telemetry\n"
        "moteus r4.11 emulation (software-in-the-loop)",
        color="#E6EDF3", fontsize=13, fontweight="bold", y=0.988
    )

    out_path = os.path.join(PLOTS_DIR, "reciprocating_actuator_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved → {out_path}\n")
    return out_path


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  HIGH-SPEED RECIPROCATING ACTUATOR SIMULATION — Task 4")
    print("  moteus | Torque Homing | Trapezoidal Profile | Watchdog")
    print("=" * 62)
    print(f"\n  Motor start : θ = {np.degrees(MOTOR_START_THETA):.1f}°  "
          f"(≈ {MOTOR_START_THETA/abs(HOMING_VEL):.1f}s visible homing journey)")
    print(f"  Homing vel  : {HOMING_VEL} rad/s  → Iq spike ≈ 2.0 A > 1.5 A threshold")
    print(f"  Stroke      : {np.degrees(TARGET_B):.0f}° ↔ {np.degrees(TARGET_A):.0f}°  "
          f"({np.degrees(TARGET_A-TARGET_B):.1f}° = {TARGET_A-TARGET_B:.2f} rad)")
    print(f"  Reciprocal  : {T_RECIPROCAL}s  @ {MAX_VEL} rad/s cruise")
    print(f"  Watchdog    : 100 ms timeout  │  Demo fault at t_recip = {WATCHDOG_DEMO_REL_T}s\n")

    # Subsystems
    motor         = BLDCMotorHS(theta0=MOTOR_START_THETA)
    vel_pd        = VelocityPD(Kp=1.5, Kd=0.01, T_max=2.0)
    current_guard = CurrentLimitGuard(threshold_a=1.5)
    watchdog      = CommandWatchdog(timeout_s=0.1)
    logger        = TelemetryLogger(path=LOG_PATH)

    # ── PHASE 1: Homing ──────────────────────────────────────────────────
    print("  [1/4] PHASE 1 — Torque-Based Homing (visible ~1 s journey) ...")
    h_log, home_pos, t_home_end = run_homing(
        motor, vel_pd, current_guard, watchdog, logger
    )
    motor.reset(theta=home_pos, omega=0.0)
    vel_pd.reset()
    current_guard.reset()
    watchdog.reset(t_home_end)

    # ── PHASE 2: Reciprocating ────────────────────────────────────────────
    print("  [2/4] PHASE 2 — Reciprocating Motion + Watchdog Demo ...")
    r_log, t_wd_start, t_wd_end = run_reciprocating(
        motor, vel_pd, current_guard, watchdog, logger, t_offset=t_home_end
    )
    logger.close()
    print(f"        Telemetry CSV → {LOG_PATH}\n")

    # ── PHASE 3: Validation ───────────────────────────────────────────────
    print("  [3/4] PHASE 3 — Validation (reciprocating phase only) ...")
    results = run_validation(r_log)

    # ── PHASE 4: Plots ────────────────────────────────────────────────────
    print("  [4/4] PHASE 4 — Generating combined telemetry plot ...")
    combined = _concat_logs(h_log, r_log)
    create_plots(combined, r_log, results, t_home_end, t_wd_start, t_wd_end)

    print("  ✓ All phases complete.")
    print("  Open plots/reciprocating_actuator_results.png to view results.\n")
