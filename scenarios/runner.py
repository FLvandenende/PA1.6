import csv
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from utils.config import Config, load_config
from utils.rng import RNG
from sensors.temp_sensor import TempSensor
from sensors.filters import hold_last, MovingAverageFilter
from controllers.onoff import OnOffThermostat
from controllers.predictive_onoff import PredictiveOnOff
from simulations.room_model import step_room
from simulations.environment import Environment
from plotting.plots import plot_timeseries, plot_error, plot_duty, plot_predictive, plot_heater


def _simulate_once(scenario: Config, seed: int) -> Tuple[Dict[str, List[float]], Dict[str, float], bool]:
    rng = RNG(seed)

    env = Environment(
        base=scenario.env.base,
        amplitude=scenario.env.amplitude,
        period_s=scenario.env.period_s,
        door_drop_C=scenario.env.door_drop_C,
        door_start_s=scenario.env.door_start_s,
        door_duration_s=scenario.env.door_duration_s,
    )

    sensor = TempSensor(
        sigma=scenario.sensor.sigma,
        bias=scenario.sensor.bias,
        dropout_prob=scenario.sensor.dropout_prob,
        rng=rng,
    )

    if getattr(scenario.controller, "type", "predictive_onoff") == "onoff":
        ctrl = OnOffThermostat(
            setpoint=scenario.controller.setpoint,
            deadband=scenario.controller.deadband,
            safety_high=scenario.controller.safety_high,
            state=0,
        )
        use_predictive = False
    else:
        ctrl = PredictiveOnOff(
            setpoint=scenario.controller.setpoint,
            deadband=scenario.controller.deadband,
            tau=scenario.controller.tau,
            safety_high=scenario.controller.safety_high,
            state=0,
        )
        use_predictive = True

    dt = scenario.sim.dt
    steps = int(scenario.sim.duration_s / dt)
    if steps < 1:
        raise ValueError("Simulation duration must be at least one time step")

    temperature = scenario.sim.init_T
    last_valid = temperature
    keys = ["t", "T_true", "T_meas", "T_out", "setpoint", "heater", "error", "T_pred", "lower", "upper"]
    log: Dict[str, List[float]] = {key: [] for key in keys}
    moving_average = MovingAverageFilter(window=5)

    for step in range(steps):
        t = step * dt
        outside_temperature = env.T_out(t)
        measured_temperature = sensor.read(temperature)
        measured_temperature = hold_last(measured_temperature, last_valid)
        if measured_temperature is None:
            measured_temperature = temperature
        last_valid = measured_temperature
        filtered_temperature = moving_average.update(measured_temperature)

        if use_predictive:
            heater = ctrl.update(filtered_temperature, dt)
            predicted_temperature = (
                ctrl.last_pred if ctrl.last_pred is not None else filtered_temperature
            )
            lower = (
                ctrl.lower_threshold
                if ctrl.lower_threshold is not None
                else ctrl.setpoint - ctrl.deadband / 2.0
            )
            upper = (
                ctrl.upper_threshold
                if ctrl.upper_threshold is not None
                else ctrl.setpoint + ctrl.deadband / 2.0
            )
        else:
            heater = ctrl.update(filtered_temperature)
            predicted_temperature = filtered_temperature
            lower = ctrl.setpoint - ctrl.deadband / 2.0
            upper = ctrl.setpoint + ctrl.deadband / 2.0

        error = ctrl.setpoint - filtered_temperature
        log["t"].append(t)
        log["T_true"].append(temperature)
        log["T_meas"].append(filtered_temperature)
        log["T_out"].append(outside_temperature)
        log["setpoint"].append(ctrl.setpoint)
        log["heater"].append(heater)
        log["error"].append(error)
        log["T_pred"].append(predicted_temperature)
        log["lower"].append(lower)
        log["upper"].append(upper)

        temperature = step_room(
            temperature,
            heater,
            outside_temperature,
            scenario.model.R,
            scenario.model.C,
            scenario.model.P,
            dt,
            scenario.model.process_sigma,
            rng,
        )

    metrics = {
        "final_temp_C": temperature,
        "min_temp_C": min(min(log["T_true"]), temperature),
        "max_temp_C": max(max(log["T_true"]), temperature),
        "mean_abs_error_C": float(np.mean(np.abs(log["error"]))),
        "heater_duty": float(np.mean(log["heater"])),
    }
    return log, metrics, use_predictive


def _write_run_log(log: Dict[str, List[float]], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(log.keys())
        writer.writerows(zip(*log.values()))


def _write_monte_carlo_results(
    base: str,
    seed: int,
    run_metrics: List[Dict[str, float]],
) -> Tuple[str, str]:
    os.makedirs(os.path.join("outputs", "logs"), exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    runs_path = os.path.join("outputs", "logs", f"{base}-monte-carlo-runs-{timestamp}.csv")
    summary_path = os.path.join("outputs", "logs", f"{base}-monte-carlo-summary-{timestamp}.csv")

    metric_names = list(run_metrics[0].keys())
    with open(runs_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["run", "seed", *metric_names])
        writer.writeheader()
        for run_index, metrics in enumerate(run_metrics):
            writer.writerow({"run": run_index + 1, "seed": seed + run_index, **metrics})

    with open(summary_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["metric", "mean", "std", "p05", "p95"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name in metric_names:
            values = np.asarray([metrics[metric_name] for metrics in run_metrics])
            writer.writerow(
                {
                    "metric": metric_name,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "p05": float(np.percentile(values, 5)),
                    "p95": float(np.percentile(values, 95)),
                }
            )

    return runs_path, summary_path


def run_scenario(scenario_path: str, runs: int = 1) -> None:
    if runs < 1:
        raise ValueError("runs must be at least 1")

    scenario = load_config(scenario_path)
    base = os.path.splitext(os.path.basename(scenario_path))[0]

    if runs > 1:
        seed = scenario.sim.seed
        run_metrics = [
            _simulate_once(scenario, seed + run_index)[1]
            for run_index in range(runs)
        ]
        runs_path, summary_path = _write_monte_carlo_results(base, seed, run_metrics)
        print(f"Wrote per-run metrics to {runs_path}")
        print(f"Wrote Monte Carlo summary to {summary_path}")
        return

    log, _, use_predictive = _simulate_once(scenario, scenario.sim.seed)
    os.makedirs(os.path.join("outputs", "logs"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("outputs", "logs")
    figure_dir = os.path.join("outputs", "figures")
    csv_path = os.path.join(log_dir, f"{base}-{timestamp}.csv")
    _write_run_log(log, csv_path)

    plot_timeseries(log, os.path.join(figure_dir, f"{base}-temps-{timestamp}.png"))
    plot_heater(log, os.path.join(figure_dir, f"{base}-heater-{timestamp}.png"))
    plot_error(log, os.path.join(figure_dir, f"{base}-error-{timestamp}.png"))
    plot_duty(log, os.path.join(figure_dir, f"{base}-duty-{timestamp}.png"))
    if use_predictive:
        plot_predictive(log, os.path.join(figure_dir, f"{base}-predictive-{timestamp}.png"))

    print(f"Wrote log to {csv_path}")
    print(f"Figures saved to {figure_dir}")
