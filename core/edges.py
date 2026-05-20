"""
edges.py — DependencyEdge
--------------------------
Directed edges in the training DAG.

Each edge represents a coaching rule:
  "Node A must come before Node B"
  "Node A enables Node B"
  "Node A is a prerequisite for Node B"

Three constraint types:
- HARD:  Must be satisfied. Violating this produces an invalid plan.
         Example: Rest must follow LongRun
- SOFT:  Should be satisfied. Violating this reduces plan quality score.
         Example: EasyRun preferred before TempoRun
- LOAD:  Weekly TRIMP budget constraint.
         Example: Total weekly load must not exceed safe ramp rate

Why explicit edges for coaching rules?
A flat weekly schedule encodes rules implicitly — the coach "just knows"
not to put two hard sessions back to back. A DAG makes every rule
explicit, auditable, and explainable.

This is the "white box modelling" requirement from Runna's job spec:
every scheduling decision has a traceable edge behind it.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ConstraintType(Enum):
    """
    Classification of dependency edges by strictness.

    HARD  → violation = invalid plan, solver must fix
    SOFT  → violation = quality penalty, solver tries to fix
    LOAD  → violation = injury risk signal, solver warns
    """
    HARD = "hard"    # Must be satisfied
    SOFT = "soft"    # Should be satisfied (quality score impact)
    LOAD = "load"    # Load budget constraint


class EdgeReason(Enum):
    """
    The coaching reason behind a dependency edge.
    Used for UI tooltips and plan explanations.

    This is what makes the plan "explainable":
    every edge has a human-readable reason.
    """
    # Recovery rules
    RECOVERY_AFTER_LONG      = "recovery_after_long_run"
    RECOVERY_AFTER_QUALITY   = "recovery_after_quality_session"
    REST_AFTER_LONG          = "rest_after_long_run"

    # Readiness rules
    FRESH_FOR_QUALITY        = "must_be_fresh_for_quality"
    BASE_BEFORE_QUALITY      = "aerobic_base_before_quality"
    EASY_BEFORE_LONG         = "easy_run_before_long_run"

    # Progression rules
    PROGRESSIVE_OVERLOAD     = "progressive_overload"
    TAPER_SEQUENCE           = "taper_sequence"

    # Load rules
    WEEKLY_LOAD_BUDGET       = "weekly_load_budget"
    RAMP_RATE_LIMIT          = "ramp_rate_limit"

    # Race rules
    RACE_READINESS           = "race_readiness"
    PRE_RACE_REST            = "pre_race_rest"


# Human-readable descriptions for each edge reason
# Used in UI tooltips and plan explanations
EDGE_DESCRIPTIONS = {
    EdgeReason.RECOVERY_AFTER_LONG:
        "Recovery run required after long run to flush fatigue",
    EdgeReason.RECOVERY_AFTER_QUALITY:
        "Easy session required after hard effort to allow adaptation",
    EdgeReason.REST_AFTER_LONG:
        "Rest day recommended after long run (>90 min)",
    EdgeReason.FRESH_FOR_QUALITY:
        "Quality session requires fresh legs — no hard effort in previous 48h",
    EdgeReason.BASE_BEFORE_QUALITY:
        "Aerobic base (easy running) must precede quality sessions",
    EdgeReason.EASY_BEFORE_LONG:
        "Easy run day before long run to arrive with adequate glycogen",
    EdgeReason.PROGRESSIVE_OVERLOAD:
        "Volume/intensity follows progressive overload principle",
    EdgeReason.TAPER_SEQUENCE:
        "Taper phase: volume reduces, sharpening workouts maintained",
    EdgeReason.WEEKLY_LOAD_BUDGET:
        "Total weekly TRIMP must stay within safe ramp rate (<10%)",
    EdgeReason.RAMP_RATE_LIMIT:
        "Week-over-week load increase capped at 10% to reduce injury risk",
    EdgeReason.RACE_READINESS:
        "Race requires sufficient fitness base (CTL ≥ target threshold)",
    EdgeReason.PRE_RACE_REST:
        "Rest day mandatory within 48h of race",
}


@dataclass
class DependencyEdge:
    """
    A directed edge from source_id → target_id in the training DAG.

    Meaning: source must be completed (or skipped with replanning)
    before target can be scheduled.

    Attributes:
        source_id:       node_id of the prerequisite workout
        target_id:       node_id of the dependent workout
        constraint_type: HARD / SOFT / LOAD
        reason:          coaching reason (for UI explanation)
        min_gap_days:    minimum rest days between source and target
        max_gap_days:    maximum days between source and target (optional)
        notes:           additional coaching context
    """
    source_id:       str
    target_id:       str
    constraint_type: ConstraintType
    reason:          EdgeReason
    min_gap_days:    int = 0
    max_gap_days:    Optional[int] = None
    notes:           str = ""

    @property
    def description(self) -> str:
        """Human-readable explanation of this constraint."""
        return EDGE_DESCRIPTIONS.get(self.reason, self.reason.value)

    @property
    def is_hard(self) -> bool:
        return self.constraint_type == ConstraintType.HARD

    @property
    def is_soft(self) -> bool:
        return self.constraint_type == ConstraintType.SOFT

    @property
    def is_load(self) -> bool:
        return self.constraint_type == ConstraintType.LOAD

    def to_dict(self) -> dict:
        """Serialise to dict for JSON API / ReactFlow edges."""
        return {
            "id":              f"{self.source_id}__{self.target_id}",
            "source":          self.source_id,
            "target":          self.target_id,
            "constraint_type": self.constraint_type.value,
            "reason":          self.reason.value,
            "description":     self.description,
            "min_gap_days":    self.min_gap_days,
            "max_gap_days":    self.max_gap_days,
            "is_hard":         self.is_hard,
            "animated":        self.is_hard,   # animated in ReactFlow = hard constraint
            "label":           self._short_label(),
        }

    def _short_label(self) -> str:
        """Short label shown on the edge in the ReactFlow UI."""
        labels = {
            EdgeReason.REST_AFTER_LONG:       "rest req.",
            EdgeReason.FRESH_FOR_QUALITY:     "fresh legs",
            EdgeReason.BASE_BEFORE_QUALITY:   "base first",
            EdgeReason.RECOVERY_AFTER_LONG:   "recovery",
            EdgeReason.RECOVERY_AFTER_QUALITY:"easy after",
            EdgeReason.PRE_RACE_REST:         "pre-race rest",
            EdgeReason.PROGRESSIVE_OVERLOAD:  "progression",
            EdgeReason.TAPER_SEQUENCE:        "taper",
        }
        return labels.get(self.reason, "")

    def __repr__(self) -> str:
        return (f"Edge({self.source_id} → {self.target_id}, "
                f"{self.constraint_type.value}, "
                f"{self.reason.value})")
