import logging
from typing import Any

import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

from src.common.config import settings

logger = logging.getLogger(__name__)


def get_production_model() -> tuple[Any, str]:
    """Return the current production LightGBM model and its version string.

    Looks up the model version pointed to by the Production alias in the MLflow
    registry and loads it. Raises RuntimeError if no production model is registered.
    """
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
    try:
        mv = client.get_model_version_by_alias(
            settings.MLFLOW_MODEL_NAME, settings.MLFLOW_PRODUCTION_ALIAS
        )
    except Exception as exc:
        raise RuntimeError(
            f"No model found with alias '{settings.MLFLOW_PRODUCTION_ALIAS}' "
            f"for '{settings.MLFLOW_MODEL_NAME}'"
        ) from exc
    model_uri = f"models:/{settings.MLFLOW_MODEL_NAME}/{mv.version}"
    model = mlflow.lightgbm.load_model(model_uri)
    logger.info("Loaded production model version %s from %s", mv.version, model_uri)
    return model, mv.version


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


def promote_to_production(version: str) -> None:
    """Point the Production alias to the given model version.

    Reassigning the alias is atomic in the registry: the old production version
    continues serving until the alias moves. Used by the initial training script
    and the deployment gate after a candidate passes the AUC improvement check.
    """
    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
    client.set_registered_model_alias(
        name=settings.MLFLOW_MODEL_NAME,
        alias=settings.MLFLOW_PRODUCTION_ALIAS,
        version=version,
    )
    logger.info(
        "Set Production alias for '%s' to version %s",
        settings.MLFLOW_MODEL_NAME,
        version,
    )
