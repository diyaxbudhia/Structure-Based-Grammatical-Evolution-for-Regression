import argparse
import os
import platform
import time
from datetime import datetime
from textwrap import dedent

import pandas as pd

MIN_RECOMMENDED_ROWS = 20000

from data_loader import build_supervised_dataset, load_series, time_split
from ge_core import GEConfig, evolve_one_run, summarize_runs


def parse_operator_rates(rate_text: str) -> tuple:
    parts = [p.strip() for p in rate_text.split(",")]
    if len(parts) != 2:
        raise ValueError("R must be 'crossover_rate,mutation_rate', e.g. '0.85,0.15'.")
    crossover = float(parts[0])
    mutation = float(parts[1])
    if not (0.0 <= crossover <= 1.0 and 0.0 <= mutation <= 1.0):
        raise ValueError("Both values in R must be within [0, 1].")
    return crossover, mutation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="COS 710 Assignment 3: Structure-Based Grammatical Evolution for electricity load regression",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=dedent(
            """
            Example quick run:
                python ge_load_forecasting.py --data "data/Residential_Energy_Dataset_UK- 2014-2020.csv" --target-col Electricity_load --start-row 0 --max-rows 6000 --mode previous_days --lag-count 7 --points-per-day 0 --runs 2 --population-size 80 --generations 30 --out-dir output_quick

            Example assignment run (minimum 10 runs):
                python ge_load_forecasting.py --data "data/Residential_Energy_Dataset_UK- 2014-2020.csv" --target-col Electricity_load --start-row 0 --max-rows 20000 --mode previous_days --lag-count 7 --points-per-day 0 --runs 10 --population-size 120 --generations 60 --out-dir output_assignment

            Alternative mode (m previous values):
                python ge_load_forecasting.py --data "data/Residential_Energy_Dataset_UK- 2014-2020.csv" --target-col Electricity_load --start-row 0 --max-rows 20000 --mode previous_values --lag-count 24 --runs 10 --population-size 120 --generations 60 --out-dir output_prev_values
            """
        ),
    )

    parser.add_argument("--data", type=str, default="data/Residential_Energy_Dataset_UK- 2014-2020.csv")
    parser.add_argument("--target-col", type=str, default="Electricity_load")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)

    parser.add_argument(
        "--mode",
        type=str,
        default="previous_days",
        choices=["previous_days", "previous_values"],
    )
    parser.add_argument("--lag-count", type=int, default=7)
    parser.add_argument("--points-per-day", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)

    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=4)

    parser.add_argument("--P", "--population-size", dest="P", type=int, default=120)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--DI", "--max-init-depth", dest="DI", type=int, default=4)
    parser.add_argument("--Dm", "--max-tree-depth", dest="Dm", type=int, default=9)
    parser.add_argument("--Du", "--mutation-max-subtree-depth", dest="Du", type=int, default=4)
    parser.add_argument("--Sg", "--global-search-generations", dest="Sg", type=int, default=30)
    parser.add_argument("--Dg", "--global-area-cutoff-depth", dest="Dg", type=int, default=9)
    parser.add_argument("--Tg", "--global-similarity-threshold", dest="Tg", type=float, default=0.75)
    parser.add_argument(
        "--Wg",
        "--no-change-window-generations",
        dest="Wg",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--DIg",
        "--transferred-global-init-depth",
        dest="DIg",
        type=int,
        default=3,
    )
    parser.add_argument("--S", "--tournament-size", dest="S", type=int, default=4)
    parser.add_argument(
        "--R",
        type=str,
        default="0.85,0.15",
        help="Crossover/mutation application rates as 'crossover,mutation'",
    )
    parser.add_argument("--preferred-min-depth", type=int, default=3)
    parser.add_argument("--preferred-max-depth", type=int, default=7)
    parser.add_argument("--elitism", type=int, default=1)
    parser.add_argument("--parsimony-lambda", type=float, default=0.001)
    parser.add_argument("--structure-depth-penalty", type=float, default=0.01)
    parser.add_argument("--hit-bound", type=float, default=0.01)

    parser.add_argument("--out-dir", type=str, default="outputs")

    return parser.parse_args()


def _system_info_text() -> str:
    return "\n".join(
        [
            f"timestamp_utc: {datetime.utcnow().isoformat()}Z",
            f"platform: {platform.platform()}",
            f"python_version: {platform.python_version()}",
            f"machine: {platform.machine()}",
            f"processor: {platform.processor()}",
            f"cpu_count_logical: {os.cpu_count()}",
        ]
    )


def main() -> None:
    args = parse_args()

    if args.runs < 1:
        raise ValueError("runs must be >= 1")
    if args.runs < 10:
        print("WARNING: Assignment report requires at least 10 runs.")

    if 0 < args.max_rows < MIN_RECOMMENDED_ROWS:
        print(
            f"WARNING: max_rows={args.max_rows} is smaller than the recommended sample size of {MIN_RECOMMENDED_ROWS}. "
            "Use --max-rows 20000 or leave max_rows=0 to use the full dataset for better generalisation."
        )

    crossover_rate, mutation_rate = parse_operator_rates(args.R)

    cfg = GEConfig(
        population_size=args.P,
        generations=args.generations,
        max_init_depth=args.DI,
        max_tree_depth=args.Dm,
        tournament_size=args.S,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism=args.elitism,
        parsimony_lambda=args.parsimony_lambda,
        structure_depth_penalty=args.structure_depth_penalty,
        global_search_generations=args.Sg,
        no_change_window_generations=args.Wg,
        global_area_cutoff_depth=args.Dg,
        global_similarity_threshold=args.Tg,
        transferred_global_init_depth=args.DIg,
        preferred_min_depth=args.preferred_min_depth,
        preferred_max_depth=args.preferred_max_depth,
    )

    series, dataset_info = load_series(
        csv_path=args.data,
        target_col=args.target_col,
        start_row=args.start_row,
        max_rows=args.max_rows,
    )

    points_per_day = args.points_per_day
    if args.mode == "previous_days" and points_per_day == 0:
        points_per_day = dataset_info.inferred_points_per_day or 96

    x, y, lags = build_supervised_dataset(
        series=series,
        mode=args.mode,
        lag_count=args.lag_count,
        points_per_day=points_per_day,
    )

    x_train, y_train, x_val, y_val, x_test, y_test = time_split(
        x,
        y,
        args.train_ratio,
        args.validation_ratio,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    run_rows = []
    all_generation_rows = []

    print("\n=== COS 710 Assignment 3: Structure-Based GE ===")
    print("Data summary")
    print(f"target: {dataset_info.target_col_used}")
    print(
        "rows: "
        f"start={dataset_info.subset_start_row}, "
        f"max_rows={dataset_info.subset_max_rows}, "
        f"selected={dataset_info.num_rows}, "
        f"original={dataset_info.original_num_rows}"
    )
    print(
        f"samples: total={len(x)}, train={len(x_train)}, validation={len(x_val)}, test={len(x_test)}"
    )
    print(f"mode: {args.mode}, lags: {lags}")
    print(
        "SBGE params: "
        f"P={cfg.population_size}, R=({cfg.crossover_rate:.2f},{cfg.mutation_rate:.2f}), "
        f"S={cfg.tournament_size}, Dm={cfg.max_tree_depth}, DI={cfg.max_init_depth}, "
        f"Sg={cfg.global_search_generations}, Dg={cfg.global_area_cutoff_depth}, "
        f"Tg={cfg.global_similarity_threshold:.2f}, Wg={cfg.no_change_window_generations}"
    )

    for run_idx in range(args.runs):
        seed = args.base_seed + run_idx
        t0 = time.perf_counter()
        _, metrics, generation_rows = evolve_one_run(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            cfg=cfg,
            seed=seed,
            hit_bound=args.hit_bound,
        )
        elapsed = time.perf_counter() - t0

        row = {
            "run": run_idx + 1,
            "seed": seed,
            "train_rmse": metrics["train_rmse"],
            "val_rmse": metrics["val_rmse"],
            "test_rmse": metrics["test_rmse"],
            "test_mae": metrics["test_mae"],
            "test_mape": metrics["test_mape"],
            "test_hit_ratio": metrics["test_hit_ratio"],
            "hit_bound_used": metrics["hit_bound_used"],
            "tree_size": metrics["tree_size"],
            "tree_depth": metrics["tree_depth"],
            "mode_switches": metrics["mode_switches"],
            "runtime_seconds": elapsed,
            "expression": metrics["expression"],
        }
        run_rows.append(row)

        for g_row in generation_rows:
            all_generation_rows.append(
                {
                    "run": run_idx + 1,
                    **g_row,
                }
            )

        print(
            f"run {row['run']:02d}/{args.runs}: "
            f"val_rmse={row['val_rmse']:.6f}, "
            f"test_rmse={row['test_rmse']:.6f}, "
            f"test_mae={row['test_mae']:.6f}, "
            f"hit_ratio={row['test_hit_ratio']:.2%}, "
            f"switches={row['mode_switches']}, "
            f"runtime={row['runtime_seconds']:.2f}s"
        )

    summary = summarize_runs(run_rows)

    per_generation_df = pd.DataFrame(all_generation_rows)
    per_generation_df = per_generation_df.rename(columns={"search_mode": "phase"})
    per_generation_df = per_generation_df[
        [
            "run",
            "generation",
            "phase",
            "average_standardized_fitness",
            "average_tree_size",
            "variety_percentage",
            "average_hit_ratio",
            "population_similarity",
        ]
    ]
    per_generation_df.to_csv(os.path.join(args.out_dir, "generation_metrics.csv"), index=False)

    pd.DataFrame(run_rows).to_csv(os.path.join(args.out_dir, "run_metrics.csv"), index=False)
    with open(os.path.join(args.out_dir, "run_summary.txt"), "w", encoding="utf-8") as out_file:
        out_file.write(_system_info_text())
        out_file.write("\n\n")
        out_file.write(dedent(str(summary)))

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
