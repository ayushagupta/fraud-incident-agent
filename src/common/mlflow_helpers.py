import logging
from typing import Any

import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

from src.common.config import settings

logger = logging.getLogger(__name__)


def get_production_model() -> tuple[Any, str]:
    """Return the current production LightGBM model and its version string.

    Queries the MLflow registry for the model version tagged as Production and
    loads it. Raises RuntimeError if no production model is registered yet.
    """
    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
    versions = client.get_latest_versions(
        settings.MLFLOW_MODEL_NAME, stages=[settings.MLFLOW_PRODUCTION_STAGE]
    )
    if not versions:
        raise RuntimeError(
            f"No model found in stage '{settings.MLFLOW_PRODUCTION_STAGE}' "
            f"for '{settings.MLFLOW_MODEL_NAME}'"
        )
    version = versions[0]
    model_uri = f"models:/{settings.MLFLOW_MODEL_NAME}/{version.version}"
    model = mlflow.lightgbm.load_model(model_uri)
    logger.info("Loaded production model version %s from %s", version.version, model_uri)
    return model, version.version


def register_new_version(run_id: str, artifact_path: str = "model") -> str:
    """Register a trained model artifact from an MLflow run into the registry.

    Creates a new model version from the given run and artifact path. Does not
    promote to Production -- the deployment gate handles that separately.
    Returns the new version string.
    """
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri, settings.MLFLOW_MODEL_NAME)
    logger.info(
        "Registered model '%s' version %s from run %s",
        settings.MLFLOW_MODEL_NAME,
        result.version,
        run_id,
    )
    return result.version
