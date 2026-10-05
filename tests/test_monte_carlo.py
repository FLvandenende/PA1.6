import csv

from scenarios.runner import run_scenario


def test_monte_carlo_writes_trials_and_summary(tmp_path, monkeypatch):
    scenario_path = tmp_path / "small_scenario.yaml"
    scenario_path.write_text(
        """\
env:
  base: 5.0
  amplitude: 2.0
  period_s: 40.0
  door_drop_C: 5.0
  door_start_s: 5.0
  door_duration_s: 10.0
sensor:
  sigma: 0.2
  bias: 0.0
  dropout_prob: 0.1
controller:
  type: onoff
  setpoint: 21.0
  deadband: 1.0
  safety_high: 26.0
model:
  R: 0.5
  C: 10000.0
  P: 200.0
  process_sigma: 0.05
sim:
  dt: 1.0
  duration_s: 20.0
  seed: 40
  init_T: 18.0
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    run_scenario(str(scenario_path), runs=3)

    log_dir = tmp_path / "outputs" / "logs"
    trials_path = next(log_dir.glob("*-monte-carlo-runs-*.csv"))
    summary_path = next(log_dir.glob("*-monte-carlo-summary-*.csv"))

    with trials_path.open(newline="", encoding="utf-8") as csv_file:
        trials = list(csv.DictReader(csv_file))
    assert [row["seed"] for row in trials] == ["40", "41", "42"]
    assert len(trials) == 3

    with summary_path.open(newline="", encoding="utf-8") as csv_file:
        summary = list(csv.DictReader(csv_file))
    assert {row["metric"] for row in summary} == {
        "final_temp_C",
        "min_temp_C",
        "max_temp_C",
        "mean_abs_error_C",
        "heater_duty",
    }
    assert all(float(row["p05"]) <= float(row["p95"]) for row in summary)
