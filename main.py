import argparse
from scenarios.runner import run_scenario

def main():
    parser = argparse.ArgumentParser(description="Run a room temperature scenario")
    parser.add_argument("--scenario", required=True, help="Path to YAML scenario file")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of Monte Carlo trials (default: 1)",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    run_scenario(args.scenario, runs=args.runs)

if __name__ == "__main__":
    main()
