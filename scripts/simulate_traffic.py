"""Replay BAF dataset rows against the serving layer to simulate production traffic.

Supports time-compressed replay and failure mode injection for testing the monitoring
and incident response pipeline.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "Base.csv"
SERVING_URL = "http://localhost:8000/predict"
TARGET_COL = "fraud_bool"
MONTH_COL = "month"
SEASONAL_FEATURE = "income"
PROGRESS_INTERVAL = 100


def load_month(month: int, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    subset = df[df[MONTH_COL] == month].copy()
    if subset.empty:
        print(f"No rows found for month {month}. Available months: {sorted(df[MONTH_COL].unique())}")
        sys.exit(1)
    subset = subset.drop(columns=[TARGET_COL, MONTH_COL])
    if limit is not None:
        subset = subset.head(limit)
    return subset.reset_index(drop=True)


def apply_injections(
    row: dict,
    null_feature: str | None,
    null_rate: float,
    unseen_column: str | None,
    unseen_value: str | None,
    seasonal_multiplier: float | None,
) -> dict:
    row = dict(row)
    if seasonal_multiplier is not None and SEASONAL_FEATURE in row:
        val = row[SEASONAL_FEATURE]
        if val is not None:
            row[SEASONAL_FEATURE] = val * seasonal_multiplier
    if null_feature is not None and random.random() < null_rate:
        row[null_feature] = None
    if unseen_column is not None:
        row[unseen_column] = unseen_value
    return row


def print_progress(sent: int, successes: int, start_time: float) -> None:
    elapsed = time.time() - start_time
    rps = sent / elapsed if elapsed > 0 else 0.0
    success_rate = successes / sent if sent > 0 else 0.0
    print(
        f"  {sent} sent | {elapsed:.1f}s elapsed | {rps:.1f} req/s | "
        f"{success_rate:.1%} success"
    )


def run(args: argparse.Namespace) -> None:
    null_feature: str | None = None
    null_rate: float = 0.0
    if args.inject_nulls:
        null_feature, rate_str = args.inject_nulls
        null_rate = float(rate_str)
        if not (0.0 <= null_rate <= 1.0):
            print(f"--inject-nulls rate must be between 0 and 1, got {null_rate}")
            sys.exit(1)

    unseen_column: str | None = None
    unseen_value: str | None = None
    if args.inject_unseen_category:
        unseen_column, unseen_value = args.inject_unseen_category

    seasonal_multiplier: float | None = args.inject_seasonal

    print(f"Loading month {args.month} from {DATA_PATH}")
    df = load_month(args.month, args.limit)
    total = len(df)
    print(f"Loaded {total} rows")

    if null_feature:
        print(f"Injecting nulls: {null_feature} at rate {null_rate:.2%}")
    if unseen_column:
        print(f"Injecting unseen category: {unseen_column}={unseen_value!r}")
    if seasonal_multiplier is not None:
        print(f"Injecting seasonal multiplier: {SEASONAL_FEATURE} x{seasonal_multiplier}")

    interval = 1.0 / args.speed
    start_time = time.time()
    successes = 0
    failures = 0

    with httpx.Client(timeout=10.0) as client:
        for i, (_, row) in enumerate(df.iterrows()):
            request_start = time.time()

            payload = apply_injections(
                row.to_dict(),
                null_feature,
                null_rate,
                unseen_column,
                unseen_value,
                seasonal_multiplier,
            )

            try:
                response = client.post(SERVING_URL, json=payload)
                if response.status_code == 200:
                    successes += 1
                else:
                    failures += 1
                    print(
                        f"  [WARN] row {i}: HTTP {response.status_code} - "
                        f"{response.text[:120]}"
                    )
            except httpx.RequestError as exc:
                failures += 1
                print(f"  [WARN] row {i}: request failed - {exc}")

            sent = i + 1
            if sent % PROGRESS_INTERVAL == 0:
                print_progress(sent, successes, start_time)

            elapsed_for_request = time.time() - request_start
            sleep_time = interval - elapsed_for_request
            if sleep_time > 0:
                time.sleep(sleep_time)

    total_elapsed = time.time() - start_time
    avg_rps = total / total_elapsed if total_elapsed > 0 else 0.0
    print(
        f"\nDone. {total} rows sent in {total_elapsed:.1f}s "
        f"({avg_rps:.1f} req/s). "
        f"Successes: {successes}, failures: {failures}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay BAF traffic against the serving layer."
    )
    parser.add_argument("--month", type=int, required=True, help="BAF month to replay")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Time compression factor. 1.0 = 1 req/s, 60 = 60 req/s.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of rows sent.",
    )
    parser.add_argument(
        "--inject-nulls",
        nargs=2,
        metavar=("FEATURE", "RATE"),
        help="Set FEATURE to null on RATE fraction of requests.",
    )
    parser.add_argument(
        "--inject-unseen-category",
        nargs=2,
        metavar=("COLUMN", "VALUE"),
        help="Replace COLUMN with VALUE on every request.",
    )
    parser.add_argument(
        "--inject-seasonal",
        type=float,
        metavar="MULTIPLIER",
        help=f"Multiply {SEASONAL_FEATURE} by MULTIPLIER on every request.",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
