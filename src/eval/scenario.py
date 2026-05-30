"""Scenario model for the eval harness.

Defines the Pydantic schema for a YAML scenario file and the loader that
reads all .yaml files from a directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, Field


class TrafficConfig(BaseModel):
    month: int
    rows: int
    speed: float


class NullsInjection(BaseModel):
    type: Literal["nulls"]
    feature: str
    rate: float


class UnseenCategoryInjection(BaseModel):
    type: Literal["unseen_category"]
    column: str
    value: str


class SeasonalInjection(BaseModel):
    type: Literal["seasonal"]
    multiplier: float


InjectionConfig = Annotated[
    Union[NullsInjection, UnseenCategoryInjection, SeasonalInjection],
    Field(discriminator="type"),
]


class SetupConfig(BaseModel):
    traffic: TrafficConfig
    injection: InjectionConfig | None = None


class ExpectedConfig(BaseModel):
    diagnosis: Literal[
        "genuine_drift",
        "pipeline_bug",
        "schema_change",
        "seasonal_pattern",
        "no_action_needed",
        "no_incident",
    ]
    action: Literal["retrain", "rollback", "alert_human", "suppress", "no_action"]


class Scenario(BaseModel):
    name: str
    description: str
    setup: SetupConfig
    expected: ExpectedConfig


def load_scenarios(directory: Path) -> list[Scenario]:
    """Load every .yaml file in directory as a Scenario, sorted by name."""
    scenarios = []
    for path in sorted(directory.glob("*.yaml")):
        with path.open() as f:
            data = yaml.safe_load(f)
        scenarios.append(Scenario.model_validate(data))
    return scenarios
