"""
serializers.py — DAG → API Response Converters
------------------------------------------------
Converts TrainingDAG objects into Pydantic response models.

The most important job here is computing ReactFlow node positions.
ReactFlow needs (x, y) coordinates for each node.

Layout strategy: Calendar grid
  x-axis = week number  (week 1 leftmost, week 12 rightmost)
  y-axis = day of week  (Monday top, Sunday bottom)

This produces a grid that looks like a real training calendar,
which makes the DAG immediately readable even to non-technical users.
"""

from typing import Dict
from core.graph import TrainingDAG
from core.nodes import WorkoutNode
from core.planner import AthleteProfile as CoreProfile
from .models import (
    PlanResponse, WorkoutNodeResponse, EdgeResponse,
    WeekSummary, NodeChangeResponse, ReplanResponse
)


# Grid spacing constants (pixels in ReactFlow canvas)
WEEK_SPACING = 220    # horizontal gap between weeks
DAY_SPACING = 90     # vertical gap between days
X_OFFSET = 80     # left margin
Y_OFFSET = 60     # top margin


def compute_node_position(week: int, day: int) -> tuple:
    """
    Compute (x, y) pixel position for a node in the ReactFlow canvas.

    Layout: calendar grid
      week 1, day 1 (Mon) → top-left
      week 12, day 7 (Sun) → bottom-right

    This makes the plan look like a training calendar,
    which is immediately intuitive for runners.
    """
    x = X_OFFSET + (week - 1) * WEEK_SPACING
    y = Y_OFFSET + (day - 1) * DAY_SPACING
    return x, y


def node_to_response(node: WorkoutNode) -> WorkoutNodeResponse:
    """Convert a WorkoutNode to its API response model."""
    x, y = compute_node_position(node.week, node.day)
    base = node.to_dict()
    return WorkoutNodeResponse(
        node_id=base["node_id"],
        workout_type=base["workout_type"],
        label=base["label"],
        week=base["week"],
        day=base["day"],
        duration_min=base["duration_min"],
        distance_km=base["distance_km"],
        notes=base["notes"],
        status=base["status"],
        trimp=base["trimp"],
        color=base["color"],
        is_quality=base["is_quality"],
        position_x=x,
        position_y=y,
    )


def get_phase(week: int, total_weeks: int) -> str:
    """Return the training phase name for a given week."""
    if week <= total_weeks * 0.25:
        return "base"
    elif week <= total_weeks * 0.67:
        return "build"
    elif week <= total_weeks * 0.83:
        return "peak"
    else:
        return "taper"


def dag_to_plan_response(dag: TrainingDAG,
                         athlete_name: str = "Athlete") -> PlanResponse:
    """
    Convert a full TrainingDAG to a PlanResponse.

    This is what the frontend receives and feeds directly
    into ReactFlow. No further transformation needed.
    """
    topo_nodes = dag.topological_sort()
    total_weeks = dag.get_week_count()

    # Convert nodes
    node_responses = [node_to_response(n) for n in topo_nodes]

    # Convert edges
    edge_responses = []
    for e in dag._edges:
        ed = e.to_dict()
        edge_responses.append(EdgeResponse(
            id=ed["id"],
            source=ed["source"],
            target=ed["target"],
            constraint_type=ed["constraint_type"],
            reason=ed["reason"],
            description=ed["description"],
            is_hard=ed["is_hard"],
            animated=ed["animated"],
            label=ed["label"],
        ))

    # Build week summaries
    week_summaries = {}
    for week in range(1, total_weeks + 1):
        week_nodes = dag.get_week_nodes(week)
        week_trimp = dag.get_week_trimp(week)
        quality_count = len(dag.get_quality_sessions_in_week(week))
        phase = get_phase(week, total_weeks)

        week_summaries[str(week)] = WeekSummary(
            week=week,
            trimp=round(week_trimp, 1),
            workouts=[node_to_response(n) for n in week_nodes],
            quality_sessions=quality_count,
            phase=phase,
        )

    return PlanResponse(
        plan_id=dag.plan_id,
        athlete_name=athlete_name,
        node_count=dag.node_count,
        edge_count=dag.edge_count,
        week_count=total_weeks,
        quality_score=dag.get_plan_quality_score(),
        nodes=node_responses,
        edges=edge_responses,
        weeks=week_summaries,
    )
