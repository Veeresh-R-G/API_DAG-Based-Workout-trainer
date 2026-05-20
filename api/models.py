"""
models.py — Pydantic Request / Response Models
------------------------------------------------
Defines the data shapes for the FastAPI endpoints.

Pydantic does two things:
1. Validates incoming request data (raises 422 if invalid)
2. Serialises outgoing response data to JSON

Why separate models from the core engine?
The core engine (nodes.py, graph.py etc.) doesn't know about
HTTP or JSON. These models are the translation layer.

This separation is called "Clean Architecture" or "Ports and Adapters" —
the core domain logic is completely independent of the delivery mechanism.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum


# ── Request Models ────────────────────────────────────────────────

class ExperienceLevel(str, Enum):
    BEGINNER     = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED     = "advanced"


class GeneratePlanRequest(BaseModel):
    """
    Request body for POST /plans/generate

    The frontend sends this when the athlete fills out
    the onboarding form (name, race distance, experience level).
    """
    name:                 str   = Field(default="Athlete", max_length=50)
    experience_level:     ExperienceLevel = ExperienceLevel.INTERMEDIATE
    weeks_to_race:        int   = Field(default=12, ge=4, le=24)
    race_distance_km:     float = Field(default=42.2, gt=0, le=200)
    current_ctl:          Optional[float] = Field(default=None, ge=0, le=150)
    max_long_run_min:     Optional[int]   = Field(default=None, ge=30, le=300)
    easy_run_pace_min_km: float = Field(default=6.0, ge=3.0, le=12.0)

    @field_validator("weeks_to_race")
    @classmethod
    def weeks_must_be_reasonable(cls, v):
        if v < 4:
            raise ValueError("Plan must be at least 4 weeks")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Rahul",
                "experience_level": "intermediate",
                "weeks_to_race": 12,
                "race_distance_km": 42.2,
                "current_ctl": 42.0,
                "easy_run_pace_min_km": 5.5,
            }
        }
    }


class MarkWorkoutRequest(BaseModel):
    """
    Optional notes when marking a workout complete or missed.
    The node_id and plan_id come from the URL path.
    """
    notes: Optional[str] = Field(default=None, max_length=500)


# ── Response Models ───────────────────────────────────────────────

class WorkoutNodeResponse(BaseModel):
    """Single workout node — consumed by ReactFlow as a node."""
    node_id:      str
    workout_type: str
    label:        str
    week:         int
    day:          int
    duration_min: float
    distance_km:  Optional[float]
    notes:        str
    status:       str
    trimp:        float
    color:        str
    is_quality:   bool

    # ReactFlow positioning
    position_x:  float = 0.0
    position_y:  float = 0.0


class EdgeResponse(BaseModel):
    """Single dependency edge — consumed by ReactFlow as an edge."""
    id:              str
    source:          str
    target:          str
    constraint_type: str
    reason:          str
    description:     str
    is_hard:         bool
    animated:        bool
    label:           str


class WeekSummary(BaseModel):
    """Summary of one training week."""
    week:             int
    trimp:            float
    workouts:         List[WorkoutNodeResponse]
    quality_sessions: int
    phase:            str


class PlanResponse(BaseModel):
    """
    Full plan response — the main API response.

    The 'nodes' and 'edges' fields are formatted for ReactFlow
    so the frontend can render them directly without transformation.
    """
    plan_id:       str
    athlete_name:  str
    node_count:    int
    edge_count:    int
    week_count:    int
    quality_score: float
    nodes:         List[WorkoutNodeResponse]
    edges:         List[EdgeResponse]
    weeks:         Dict[str, WeekSummary]


class NodeChangeResponse(BaseModel):
    node_id:  str
    action:   str
    reason:   str
    old_week: Optional[int]
    old_day:  Optional[int]
    new_week: Optional[int]
    new_day:  Optional[int]


class ReplanResponse(BaseModel):
    """
    Response after marking a workout complete or missed.
    Contains the updated plan + a summary of what changed.
    """
    plan_id:             str
    triggered_by:        str
    affected_count:      int
    plan_quality_before: float
    plan_quality_after:  float
    warnings:            List[str]
    changes:             List[NodeChangeResponse]
    updated_plan:        PlanResponse


class HealthResponse(BaseModel):
    status:  str
    version: str
    message: str
