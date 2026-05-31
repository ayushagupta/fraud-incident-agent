"""Manual rollback: re-point the Production alias to a prior model version."""

from __future__ import annotations

import argparse
import logging

import mlflow

from src.common.config import settings
from src.common.mlflow_helpers import promote_to_production

logger = logging.getLogger(__name__)


def rollback_to(version: str) -> None:
    """Re-point the Production alias to the given prior model version.

    No evaluation is performed. The caller is responsible for choosing a
    known-good version. Used after a production incident when the normal
    gate path is too slow.
    """
    promote_to_production(version)
    logger.info("Production alias for '%s' rolled back to version %s", settings.MLFLOW_MODEL_NAME, version)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roll back the production model alias to a prior version."
    )
    parser.add_argument("version", help="MLflow model version to promote to Production.")
    args = parser.parse_args()

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    rollback_to(args.version)
    print(f"Production alias now points to version {args.version}")


if __name__ == "__main__":
    main()
