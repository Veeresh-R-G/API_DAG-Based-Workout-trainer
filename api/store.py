"""
store.py — In-Memory Plan Store
---------------------------------
Manages active training plans during the API session.

Why in-memory?
For a portfolio project, an in-memory dict is clean and sufficient.
The frontend generates a plan, interacts with it, and the state
lives in the server's memory for the session.

Production upgrade path:
Replace with Redis (fast, persistent) or PostgreSQL (relational).
The store interface stays the same — only the backend changes.
This is the "Repository Pattern" — the API doesn't know or care
where plans are stored.
"""

from typing import Dict, Optional
from core.graph import TrainingDAG


class PlanStore:
    """
    Simple in-memory store for active training plans.
    Maps plan_id → TrainingDAG instance.
    """

    def __init__(self):
        self._plans: Dict[str, TrainingDAG] = {}

    def save(self, dag: TrainingDAG) -> None:
        """Save or update a plan."""
        self._plans[dag.plan_id] = dag

    def get(self, plan_id: str) -> Optional[TrainingDAG]:
        """Retrieve a plan by ID. Returns None if not found."""
        return self._plans.get(plan_id)

    def delete(self, plan_id: str) -> bool:
        """Delete a plan. Returns True if it existed."""
        if plan_id in self._plans:
            del self._plans[plan_id]
            return True
        return False

    def list_ids(self):
        """All active plan IDs."""
        return list(self._plans.keys())

    @property
    def count(self) -> int:
        return len(self._plans)


# Global store instance — shared across all API requests
plan_store = PlanStore()
