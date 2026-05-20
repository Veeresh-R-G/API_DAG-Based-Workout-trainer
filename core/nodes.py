"""
nodes.py — WorkoutNode
----------------------
The fundamental unit of the training DAG.

Every workout in the plan is a node with:
- A type (easy run, tempo, long run, rest, etc.)
- Duration and intensity (used for TRIMP / load calculation)
- Week and day assignment
- A status (pending, complete, missed, skipped)

Why an enum for WorkoutType?
Because we need to encode coaching rules that are TYPE-specific.
"Rest must follow LongRun" is a rule about types, not about
specific instances. This lets the constraint engine work
generically across any plan.

Why a status enum?
Dynamic replanning requires knowing which nodes have been
completed and which have been missed. Status changes propagate
through the graph — a missed node may invalidate downstream
dependencies.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import uuid


class WorkoutType(Enum):
    """
    All possible workout types in a running training plan.

    Grouped by physiological purpose:
    - AEROBIC BASE:  EasyRun, RecoveryRun, LongRun
    - QUALITY:       TempoRun, IntervalRun, HillRun
    - RECOVERY:      Rest, CrossTrain, Strength
    - EVENT:         Race, TimeTrialRun
    """
    # ── Aerobic base ──────────────────────────────────────────────
    EASY_RUN      = "easy_run"       # Zone 2, conversational pace
    RECOVERY_RUN  = "recovery_run"   # Very easy, day after hard session
    LONG_RUN      = "long_run"       # Weekly long run, aerobic endurance

    # ── Quality sessions ──────────────────────────────────────────
    TEMPO_RUN     = "tempo_run"      # Lactate threshold pace (~1hr race pace)
    INTERVAL_RUN  = "interval_run"   # VO2max efforts (e.g. 5×1km)
    HILL_RUN      = "hill_run"       # Hill repeats, strength + speed

    # ── Recovery / non-running ────────────────────────────────────
    REST          = "rest"           # Full rest day
    CROSS_TRAIN   = "cross_train"    # Bike, swim, yoga — active recovery
    STRENGTH      = "strength"       # Gym / bodyweight session

    # ── Events ────────────────────────────────────────────────────
    RACE          = "race"           # Target race
    TIME_TRIAL    = "time_trial"     # In-training benchmark effort


class NodeStatus(Enum):
    """
    Lifecycle status of a workout node.

    PENDING  → default, not yet due
    COMPLETE → athlete did the workout ✅
    MISSED   → athlete skipped / couldn't do it ❌
    SKIPPED  → replanner deliberately removed it (different from missed)
    LOCKED   → cannot be moved or modified (e.g. race day)
    """
    PENDING  = "pending"
    COMPLETE = "complete"
    MISSED   = "missed"
    SKIPPED  = "skipped"
    LOCKED   = "locked"


# Intensity zones mapped to multipliers for TRIMP calculation
# Based on Banister (1991) HR reserve zones
INTENSITY_MULTIPLIER = {
    WorkoutType.RECOVERY_RUN: 0.65,
    WorkoutType.EASY_RUN:     0.72,
    WorkoutType.CROSS_TRAIN:  0.70,
    WorkoutType.STRENGTH:     0.60,
    WorkoutType.LONG_RUN:     0.75,
    WorkoutType.HILL_RUN:     0.85,
    WorkoutType.TEMPO_RUN:    0.88,
    WorkoutType.INTERVAL_RUN: 0.95,
    WorkoutType.TIME_TRIAL:   0.93,
    WorkoutType.RACE:         1.00,
    WorkoutType.REST:         0.00,
}

# Human-readable labels for UI display
WORKOUT_LABELS = {
    WorkoutType.EASY_RUN:     "Easy Run",
    WorkoutType.RECOVERY_RUN: "Recovery Run",
    WorkoutType.LONG_RUN:     "Long Run",
    WorkoutType.TEMPO_RUN:    "Tempo Run",
    WorkoutType.INTERVAL_RUN: "Interval Run",
    WorkoutType.HILL_RUN:     "Hill Run",
    WorkoutType.REST:         "Rest Day",
    WorkoutType.CROSS_TRAIN:  "Cross Train",
    WorkoutType.STRENGTH:     "Strength",
    WorkoutType.RACE:         "Race 🏁",
    WorkoutType.TIME_TRIAL:   "Time Trial",
}

# Colour coding for the ReactFlow UI
NODE_COLORS = {
    WorkoutType.EASY_RUN:     "#4CAF50",   # green
    WorkoutType.RECOVERY_RUN: "#81C784",   # light green
    WorkoutType.LONG_RUN:     "#2196F3",   # blue
    WorkoutType.TEMPO_RUN:    "#FF9800",   # orange
    WorkoutType.INTERVAL_RUN: "#F44336",   # red
    WorkoutType.HILL_RUN:     "#FF5722",   # deep orange
    WorkoutType.REST:         "#9E9E9E",   # grey
    WorkoutType.CROSS_TRAIN:  "#00BCD4",   # cyan
    WorkoutType.STRENGTH:     "#9C27B0",   # purple
    WorkoutType.RACE:         "#FFD700",   # gold
    WorkoutType.TIME_TRIAL:   "#FFC107",   # amber
}


@dataclass
class WorkoutNode:
    """
    A single workout in the training plan.

    This is a node in the DAG. The graph engine adds directed
    edges between nodes to represent coaching dependencies.

    Attributes:
        node_id:      Unique identifier (auto-generated)
        workout_type: What kind of session this is
        week:         Which week of the plan (1-indexed)
        day:          Which day of the week (1=Mon, 7=Sun)
        duration_min: Planned duration in minutes
        distance_km:  Planned distance in km (optional)
        notes:        Coaching notes / session description
        status:       Current lifecycle status
        trimp:        Training Impulse (computed from type + duration)
    """
    workout_type: WorkoutType
    week:         int
    day:          int                    # 1=Mon ... 7=Sun
    duration_min: float
    distance_km:  Optional[float] = None
    notes:        str = ""
    status:       NodeStatus = NodeStatus.PENDING
    node_id:      str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self):
        """Validate inputs and compute derived fields."""
        if not 1 <= self.week <= 52:
            raise ValueError(f"Week must be 1-52, got {self.week}")
        if not 1 <= self.day <= 7:
            raise ValueError(f"Day must be 1-7, got {self.day}")
        if self.duration_min < 0:
            raise ValueError(f"Duration must be non-negative")

    @property
    def trimp(self) -> float:
        """
        Training Impulse — how much physiological stress this
        workout produces. Used for load constraint checking.

        TRIMP = duration × intensity_multiplier × scaling_factor
        Roughly comparable to Banister's HR-based TRIMP.
        """
        multiplier = INTENSITY_MULTIPLIER.get(self.workout_type, 0.7)
        return self.duration_min * multiplier * 0.8

    @property
    def label(self) -> str:
        """Human-readable label for UI display."""
        return WORKOUT_LABELS.get(self.workout_type, self.workout_type.value)

    @property
    def color(self) -> str:
        """Hex color for ReactFlow node rendering."""
        return NODE_COLORS.get(self.workout_type, "#9E9E9E")

    @property
    def is_quality_session(self) -> bool:
        """True if this is a hard / high-intensity session."""
        return self.workout_type in {
            WorkoutType.TEMPO_RUN,
            WorkoutType.INTERVAL_RUN,
            WorkoutType.HILL_RUN,
            WorkoutType.RACE,
            WorkoutType.TIME_TRIAL,
        }

    @property
    def is_complete(self) -> bool:
        return self.status == NodeStatus.COMPLETE

    @property
    def is_missed(self) -> bool:
        return self.status == NodeStatus.MISSED

    @property
    def is_pending(self) -> bool:
        return self.status == NodeStatus.PENDING

    def mark_complete(self) -> None:
        if self.status == NodeStatus.LOCKED:
            raise ValueError(f"Cannot mark locked node {self.node_id} as complete")
        self.status = NodeStatus.COMPLETE

    def mark_missed(self) -> None:
        if self.status == NodeStatus.LOCKED:
            raise ValueError(f"Cannot mark locked node {self.node_id} as missed")
        self.status = NodeStatus.MISSED

    def to_dict(self) -> dict:
        """Serialise to dict for JSON API response."""
        return {
            "node_id":      self.node_id,
            "workout_type": self.workout_type.value,
            "label":        self.label,
            "week":         self.week,
            "day":          self.day,
            "duration_min": self.duration_min,
            "distance_km":  self.distance_km,
            "notes":        self.notes,
            "status":       self.status.value,
            "trimp":        round(self.trimp, 1),
            "color":        self.color,
            "is_quality":   self.is_quality_session,
        }

    def __repr__(self) -> str:
        return (f"WorkoutNode({self.label}, "
                f"W{self.week}D{self.day}, "
                f"{self.duration_min:.0f}min, "
                f"{self.status.value})")
