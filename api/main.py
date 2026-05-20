"""
main.py — FastAPI Application
------------------------------
The HTTP API layer connecting the Python core engine
to the React frontend.

Endpoints:
  GET  /                          Health check
  POST /plans/generate            Generate a new training plan
  GET  /plans/{plan_id}           Get current plan state
  GET  /plans/{plan_id}/week/{w}  Get one week's workouts
  POST /plans/{plan_id}/complete/{node_id}  Mark workout done
  POST /plans/{plan_id}/miss/{node_id}      Mark workout missed
  DELETE /plans/{plan_id}         Delete a plan

Architecture:
  Request → Pydantic validation → Core engine → Pydantic serialisation → Response

The API layer knows nothing about graph algorithms.
The core engine knows nothing about HTTP.
Clean separation.
"""

from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware

from core.planner import PlanGenerator
from core.planner import AthleteProfile as CoreProfile
from core.replanner import DynamicReplanner
from core.nodes import NodeStatus

from api.models import (
    GeneratePlanRequest, MarkWorkoutRequest,
    PlanResponse, ReplanResponse, WeekSummary,
    HealthResponse, NodeChangeResponse,
)
from api.store import plan_store
from api.serializers import dag_to_plan_response, node_to_response


# ── App initialisation ────────────────────────────────────────────

app = FastAPI(
    title="DAG Training Planner API",
    description="Adaptive running training plans modelled as Directed Acyclic Graphs",
    version="1.0.0",
    docs_url="/docs",     # Swagger UI at /docs
    redoc_url="/redoc",    # ReDoc at /redoc
)

# CORS — allow the Next.js frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001",
                   "https://*.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ────────────────────────────────────────────────────────

def _get_plan_or_404(plan_id: str):
    """Retrieve plan from store or raise 404."""
    dag = plan_store.get(plan_id)
    if dag is None:
        raise HTTPException(
            status_code=404,
            detail=f"Plan '{plan_id}' not found. Generate a plan first."
        )
    return dag


def _build_core_profile(req: GeneratePlanRequest) -> CoreProfile:
    """Map API request to core AthleteProfile."""
    # Use preset profiles as base, override with request values
    base_map = {
        "beginner":     CoreProfile.beginner(),
        "intermediate": CoreProfile.intermediate(),
        "advanced":     CoreProfile.advanced(),
    }
    profile = base_map[req.experience_level.value]
    profile.name = req.name
    profile.weeks_to_race = req.weeks_to_race
    profile.race_distance_km = req.race_distance_km
    profile.easy_run_pace_min_km = req.easy_run_pace_min_km

    # Override CTL if provided
    if req.current_ctl is not None:
        profile.current_ctl = req.current_ctl

    # Override max long run if provided
    if req.max_long_run_min is not None:
        profile.max_long_run_min = req.max_long_run_min

    return profile


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        message=f"DAG Training Planner API — {plan_store.count} active plans",
    )


@app.post(
    "/plans/generate",
    response_model=PlanResponse,
    tags=["Plans"],
    summary="Generate a new training plan",
    description="""
    Generate a 12-week marathon training plan as a DAG.
    
    The response includes:
    - `nodes`: workout nodes formatted for ReactFlow
    - `edges`: dependency edges formatted for ReactFlow
    - `weeks`: week-by-week summary with TRIMP loads
    - `plan_id`: use this for subsequent requests
    """,
)
def generate_plan(request: GeneratePlanRequest) -> PlanResponse:
    """
    Generate a new training plan DAG from an athlete profile.

    Steps:
    1. Map request to core AthleteProfile
    2. PlanGenerator builds the DAG (nodes + edges)
    3. Store the DAG
    4. Serialise to PlanResponse (ReactFlow-ready JSON)
    """
    profile = _build_core_profile(request)
    dag = PlanGenerator(profile).generate()
    plan_store.save(dag)

    return dag_to_plan_response(dag, athlete_name=request.name)


@app.get(
    "/plans/{plan_id}",
    response_model=PlanResponse,
    tags=["Plans"],
    summary="Get current plan state",
)
def get_plan(plan_id: str = Path(description="Plan ID from /generate")) -> PlanResponse:
    """Retrieve the current state of a training plan."""
    dag = _get_plan_or_404(plan_id)
    return dag_to_plan_response(dag)


@app.get(
    "/plans/{plan_id}/week/{week}",
    response_model=WeekSummary,
    tags=["Plans"],
    summary="Get workouts for a specific week",
)
def get_week(
    plan_id: str = Path(description="Plan ID"),
    week:    int = Path(ge=1, le=52, description="Week number (1-indexed)"),
) -> WeekSummary:
    """Get all workouts for a specific week with load summary."""
    dag = _get_plan_or_404(plan_id)

    if week > dag.get_week_count():
        raise HTTPException(
            status_code=404,
            detail=f"Week {week} does not exist. Plan has {dag.get_week_count()} weeks."
        )

    from api.serializers import get_phase
    week_nodes = dag.get_week_nodes(week)

    return WeekSummary(
        week=week,
        trimp=round(dag.get_week_trimp(week), 1),
        workouts=[node_to_response(n) for n in week_nodes],
        quality_sessions=len(dag.get_quality_sessions_in_week(week)),
        phase=get_phase(week, dag.get_week_count()),
    )


@app.post(
    "/plans/{plan_id}/complete/{node_id}",
    response_model=ReplanResponse,
    tags=["Workouts"],
    summary="Mark a workout as completed",
)
def mark_complete(
    plan_id: str = Path(description="Plan ID"),
    node_id: str = Path(description="Node ID to mark complete"),
    body:    MarkWorkoutRequest = MarkWorkoutRequest(),
) -> ReplanResponse:
    """
    Mark a workout as completed.

    Updates the node status to COMPLETE and returns a summary
    of any downstream effects (nodes that are now unlocked).
    """
    dag = _get_plan_or_404(plan_id)

    try:
        replanner = DynamicReplanner(dag)
        summary = replanner.mark_complete(node_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Persist updated plan
    plan_store.save(dag)

    return ReplanResponse(
        plan_id=plan_id,
        triggered_by=summary.triggered_by,
        affected_count=summary.affected_count,
        plan_quality_before=summary.plan_quality_before,
        plan_quality_after=summary.plan_quality_after,
        warnings=summary.warnings,
        changes=[
            NodeChangeResponse(
                node_id=c.node_id,
                action=c.action.value,
                reason=c.reason,
                old_week=c.old_week,
                old_day=c.old_day,
                new_week=c.new_week,
                new_day=c.new_day,
            )
            for c in summary.changes
        ],
        updated_plan=dag_to_plan_response(dag),
    )


@app.post(
    "/plans/{plan_id}/miss/{node_id}",
    response_model=ReplanResponse,
    tags=["Workouts"],
    summary="Mark a workout as missed — triggers replanning",
    description="""
    Mark a workout as missed and trigger dynamic replanning.
    
    The replanner:
    1. Marks the node MISSED
    2. Finds all downstream nodes with hard dependencies on this node
    3. Tries to reschedule them to the next available slot
    4. If rescheduling fails, marks them SKIPPED
    5. Returns a full summary of all changes
    
    The `warnings` field highlights quality sessions that were dropped.
    """,
)
def mark_missed(
    plan_id: str = Path(description="Plan ID"),
    node_id: str = Path(description="Node ID to mark missed"),
    body:    MarkWorkoutRequest = MarkWorkoutRequest(),
) -> ReplanResponse:
    """
    Mark a workout as missed and trigger dynamic replanning.
    """
    dag = _get_plan_or_404(plan_id)

    try:
        replanner = DynamicReplanner(dag)
        summary = replanner.mark_missed(node_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Persist updated plan
    plan_store.save(dag)

    return ReplanResponse(
        plan_id=plan_id,
        triggered_by=summary.triggered_by,
        affected_count=summary.affected_count,
        plan_quality_before=summary.plan_quality_before,
        plan_quality_after=summary.plan_quality_after,
        warnings=summary.warnings,
        changes=[
            NodeChangeResponse(
                node_id=c.node_id,
                action=c.action.value,
                reason=c.reason,
                old_week=c.old_week,
                old_day=c.old_day,
                new_week=c.new_week,
                new_day=c.new_day,
            )
            for c in summary.changes
        ],
        updated_plan=dag_to_plan_response(dag),
    )


@app.delete(
    "/plans/{plan_id}",
    tags=["Plans"],
    summary="Delete a training plan",
)
def delete_plan(
    plan_id: str = Path(description="Plan ID to delete"),
):
    """Delete a plan from the store."""
    deleted = plan_store.delete(plan_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"Plan '{plan_id}' not found")
    return {"message": f"Plan {plan_id} deleted", "status": "ok"}


@app.get(
    "/plans",
    tags=["Plans"],
    summary="List all active plan IDs",
)
def list_plans():
    """List all active plan IDs in the store."""
    return {
        "plan_ids": plan_store.list_ids(),
        "count":    plan_store.count,
    }
