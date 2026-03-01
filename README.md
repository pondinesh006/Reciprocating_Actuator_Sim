# High-Speed Reciprocating Actuator Simulation

A **software-in-the-loop** simulation of a **moteus r4.11** brushless DC motor controller driving a high-speed reciprocating actuator. Implements three distinct behaviours — torque-based homing, trapezoidal velocity profile reciprocation, and a deliberate watchdog fault — and produces a 5-panel telemetry plot plus a full CSV log.

---

## Demonstrated Behaviours

| Phase | Description |
|-------|-------------|
| **1 — Torque Homing** | Motor starts at θ = 2.0 rad. Driven at −2.0 rad/s toward hard-stop at θ = 0. Contact detected when Iq spikes above 1.5 A (~1 s journey). |
| **2 — Reciprocating Motion** | Trapezoidal velocity profile, 0.0 ↔ 2.0 rad at 5 rad/s for 10 s (36+ strokes). Watchdog is petted every cycle → 0 natural faults. |
| **3 — Watchdog Fault Demo** | At t_recip = 7 s, commands are deliberately paused for 250 ms. The watchdog fires, torque is zeroed, motor coasts to a stop, then resumes. |

---

## Project Structure

```
Reciprocating_Actuator/
├── main.py                  # Simulation orchestrator — 4 phases
├── motor.py                 # BLDCMotorHS: physical dynamics model
├── velocity_controller.py   # VelocityPD + trapezoidal_profile + apply_hard_stop
├── watchdog.py              # CommandWatchdog + CurrentLimitGuard
├── telemetry.py             # TelemetryLogger — CSV writer + numpy buffer
├── log.csv                  # Full telemetry (homing + reciprocating)
└── plots/
    └── reciprocating_actuator_results.png   # 5-panel engineering timeline
```

---

## Module Details

### `main.py` — Simulation Orchestrator

Runs four sequential phases at a 1 kHz timestep (`DT = 0.001 s`):

| Phase | Function | Description |
|-------|----------|-------------|
| 1 | `run_homing()` | Drives motor to hard-stop; exits on Iq spike |
| 2 | `run_reciprocating()` | 10 s reciprocation with embedded watchdog demo |
| 3 | `run_validation()` | Computes 6 pass/fail metrics on the reciprocating log |
| 4 | `create_plots()` | Generates 5-panel dark-theme telemetry PNG |

**Key simulation parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DT` | 1 ms | Timestep (1 kHz control loop) |
| `MOTOR_START_THETA` | 2.0 rad | Initial shaft position |
| `HOMING_VEL` | −2.0 rad/s | Homing drive velocity |
| `TARGET_A / TARGET_B` | 2.0 / 0.0 rad | Stroke endpoints |
| `MAX_VEL` | 5.0 rad/s | Cruise speed |
| `MAX_ACC` | 20.0 rad/s² | Acceleration limit |
| `WATCHDOG_DEMO_REL_T` | 7.0 s | When command silence starts |
| `WATCHDOG_GAP_S` | 250 ms | Command gap (> 100 ms timeout) |

**Validation metrics (auto-reported):**

| Metric | Pass Criterion |
|--------|---------------|
| Mean endpoint tracking error | < 0.05 rad |
| Peak phase current Iq | < 2.01 A |
| Minimum bus voltage | > 22.0 V |
| Max winding temperature | < 100.0 °C |
| Watchdog faults triggered | ≥ 1 |
| Stroke reversals | ≥ 10 |

---

### `motor.py` — BLDCMotorHS

Discrete-time BLDC motor physics model. Each call to `step(T_input, dt)` executes a 9-step pipeline:

```
1. Saturate torque          T_motor = clip(T_input, ±T_max)
2. Compute drive current    Iq = T_motor / Kt
3. Bus voltage sag          V_bus = 24.0 − 0.10 · |Iq|
4. Coulomb friction         T_fric = T_friction · tanh(ω / 0.05)
5. Back-EMF derating        factor = 1 − (Ke·|ω|) / V_bus
6. Newton-Euler             α = (T_motor·factor − B·ω − T_fric) / J
7. Forward Euler            ω += α·dt
8. Velocity clamp           ω = clip(ω, ±15 rad/s)
9. Thermal model            T_winding += (Iq²·R_ph·R_thermal − ΔT) / C_thermal · dt
```

**Motor constants:**

| Symbol | Value | Description |
|--------|-------|-------------|
| `Kt` | 1.0 N·m/A | Torque constant |
| `Ke` | 1.0 V·s/rad | Back-EMF constant |
| `R_ph` | 0.5 Ω | Phase resistance |
| `J` | 2×10⁻⁴ kg·m² | Moment of inertia |
| `B` | 5×10⁻³ N·m·s/rad | Viscous damping |
| `T_friction` | 0.01 N·m | Coulomb friction |
| `T_max` | 2.0 N·m | Torque saturation (moteus `maximum_torque`) |
| `V_BUS_NOMINAL` | 24.0 V | Nominal bus voltage |

**Hardware analogy:** `BLDCMotorHS.step()` ↔ `moteus.set_position(velocity=v, maximum_torque=2.0)`

---

### `velocity_controller.py` — Motion Profile & PD Controller

Two-layer velocity control mirroring the moteus firmware velocity loop.

#### `trapezoidal_profile(theta, target, max_vel, max_acc, dt)`

Computes a velocity setpoint shaped into three zones:

```
Braking zone  (|error| ≤ s_brake):  vel = sign(error) · √(2·a·|error|)
Cruise zone   (|error| >  s_brake):  vel = sign(error) · max_vel

where:  s_brake = max_vel² / (2·max_acc)
```

#### `VelocityPD`

PD controller mapping velocity error → torque command:

```
T = Kp·(ω_target − ω_actual) − Kd·(dω/dt)
```

Derivative is computed **on measurement** (no derivative kick on setpoint changes).

| Parameter | Value |
|-----------|-------|
| `Kp` | 1.5 N·m·s/rad |
| `Kd` | 0.01 N·m·s²/rad |
| `T_max` | ±2.0 N·m |

#### `apply_hard_stop(theta, omega)`

Clamps shaft angle to `[0.0, 2.5]` rad. On contact, velocity is zeroed (perfectly inelastic collision, `restitution=0`).

---

### `watchdog.py` — Safety Guards

#### `CommandWatchdog`

Mirrors the moteus 100 ms command watchdog. Call `pet(sim_time)` each control cycle; call `check(sim_time)` to detect timeout.

| Method | Description |
|--------|-------------|
| `pet(t)` | Records last command time, clears fault |
| `check(t)` | Returns `True` if `t − last_command > 100 ms` |
| `reset(t)` | Clears fault and restarts timer |

**Hardware mapping:** `pet()` ↔ `await controller.set_position(...)` ; fault ↔ `controller.set_stop()`

#### `CurrentLimitGuard`

Monitors Iq every step; triggers on `|Iq| > 1.5 A`.

Used exclusively during homing — the current spike on hard-stop contact signals `homing_complete`.

| Method | Description |
|--------|-------------|
| `check(iq)` | Returns `True` if threshold exceeded |
| `reset()` | Clears fault and peak Iq tracker |

---

### `telemetry.py` — TelemetryLogger

Real-time CSV writer with an in-memory numpy buffer. Matches the moteus telemetry column layout.

**CSV columns:**

| Column | Unit | Description |
|--------|------|-------------|
| `t` | s | Simulation time |
| `target` | rad | Reference position (homing target or stroke endpoint) |
| `actual` | rad | Motor shaft angle |
| `velocity` | rad/s | Angular velocity |
| `voltage` | V | Bus voltage |
| `temp` | °C | Winding temperature |
| `iq` | A | Phase current |
| `torque` | N·m | Applied torque |
| `watchdog_fault` | 0/1 | 1 when watchdog is active |
| `homing_phase` | 0/1 | 1 during homing, 0 during reciprocation |

---

## Requirements

```
python >= 3.8
numpy
matplotlib
```

Install dependencies:

```bash
pip install numpy matplotlib
```

---

## Running the Simulation

```bash
python main.py
```

**Progress output:**

```
[1/4] PHASE 1 — Torque-Based Homing ...
  [HOMING] Motor at θ = 114.6° → driving to hard-stop at θ = 0° ...
  [HOMING] ⚡ Hard-stop DETECTED at θ = 0.00°   |Iq| = 2.000 A   (t = 1.003s)

[2/4] PHASE 2 — Reciprocating Motion + Watchdog Demo ...
  [WD]  [WD] Watchdog FAULT at t = 8.003s  (gap = 101.0 ms > 100 ms)

[3/4] PHASE 3 — Validation ...
  ✓ PASS  Mean endpoint tracking error  [rad]  = 0.0xxxx
  ✓ PASS  Peak phase current Iq         [A]    = 2.0000
  ...
  Overall: ALL TESTS PASSED ✓

[4/4] PHASE 4 — Generating combined telemetry plot ...
  Plot saved → plots/reciprocating_actuator_results.png
```

**Output files:**

| File | Description |
|------|-------------|
| `log.csv` | Full telemetry — ~13,000 rows (homing + 10 s reciprocation at 1 kHz) |
| `plots/reciprocating_actuator_results.png` | 5-panel timeline (position, velocity, Iq, torque, voltage/temp) |

---

## Output Plot — 5 Panels

| Panel | Signal | Key Features |
|-------|--------|-------------|
| 1 — Position Tracking | Target θ vs Actual θ | Homing descent, reciprocating square-wave profile |
| 2 — Angular Velocity | ω [rad/s] | Trapezoidal ramp-up/coast/deceleration cycles |
| 3 — Phase Current Iq | Iq [A] | ⚡ Homing spike > 1.5 A threshold clearly visible |
| 4 — Applied Torque | τ [N·m] | Torque → 0 at watchdog fault (coast-down) |
| 5 — Voltage & Temp | V_bus [V] + T_winding [°C] | Voltage sag under load; thermal rise |

Three annotated markers on all panels:
- 🟢 **"Homing complete ✓"** — vertical line at end of homing phase
- 🔴 **"[WD] Watchdog FAULT"** — vertical line at fault onset
- 🟡 **"Resume"** — vertical line when commands restart

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: TORQUE HOMING                                         │
│  VelocityPD(vel_cmd = -2 rad/s)                                 │
│      → torque command → BLDCMotorHS.step()                      │
│      → apply_hard_stop() clamps θ=0 → Iq spikes                │
│      → CurrentLimitGuard detects |Iq| > 1.5 A → DONE           │
├─────────────────────────────────────────────────────────────────┤
│  PHASE 2: RECIPROCATING                                         │
│  trapezoidal_profile(θ, target)  →  vel_cmd                     │
│      → VelocityPD  →  torque  →  BLDCMotorHS.step()            │
│      → CommandWatchdog.pet() every step → no fault              │
│      At t_recip=7s: pet() STOPS for 250 ms                      │
│      → CommandWatchdog.check() fires → torque=0 → coast         │
│      After 250 ms: pet() resumes → recovery                     │
├─────────────────────────────────────────────────────────────────┤
│  PHASE 3: VALIDATION    6 automated pass/fail checks            │
│  PHASE 4: PLOTTING      5-panel dark-theme telemetry figure     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Tips

| Parameter | Location | Effect |
|-----------|----------|--------|
| `HOMING_VEL` | `main.py` | Faster → larger Iq spike; must exceed 1.5 A threshold |
| `MAX_VEL` / `MAX_ACC` | `main.py` | Controls stroke speed and sharpness of trapezoidal ramps |
| `WATCHDOG_DEMO_REL_T` | `main.py` | When (relative to reciprocation start) the fault is injected |
| `Kp` / `Kd` | `main.py` (VelocityPD init) | Tune torque response to velocity error |
| `T_max` | `motor.py` | Mirrors moteus `maximum_torque`; also caps Iq |
| Thermal constants | `motor.py` | `thermal_R` / `thermal_C` set heating/cooling rate |
| `threshold_a` | `watchdog.py` | Homing detection sensitivity; default 1.5 A |
