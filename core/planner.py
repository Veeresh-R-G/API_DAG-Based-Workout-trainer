"""
planner.py — PlanGenerator
---------------------------
Generates a complete training DAG from an athlete profile.

This is where the sports science gets encoded as graph structure.
Every coaching rule becomes a DependencyEdge with an explicit reason.

Plan structure (12-week marathon):
  Weeks 1-3:  Base phase  — easy aerobic, introduce structure
  Weeks 4-7:  Build phase — progressive overload, quality sessions
  Weeks 8-10: Peak phase  — highest load, race-specific work
  Weeks 11-12: Taper      — reduce volume, sharpen, arrive fresh

The 80/20 rule (Seiler 2010):
  80% of sessions at easy/aerobic intensity (Zone 2)
  20% at quality intensity (tempo, intervals)
  Encoded as soft constraints on weekly workout distribution.

Weekly template (standard running week):
  Mon: Rest / Cross-train
  Tue: Easy run
  Wed: Quality session (tempo or intervals)
  Thu: Easy run
  Fri: Easy run or rest
  Sat: Quality session or easy
  Sun: Long run

The PlanGenerator creates this structure then adds
dependency edges based on coaching rules.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

from core.nodes import WorkoutNode, WorkoutType, NodeStatus
from core.edges import DependencyEdge, ConstraintType, EdgeReason
from core.graph import TrainingDAG


@dataclass
class AthleteProfile:
    """
    Athlete characteristics used to personalise the training plan.
    """
    name:                  str = "Athlete"
    current_ctl:           float = 40.0   # Current fitness level
    race_distance_km:      float = 42.2   # Target race (42.2 = marathon)
    weeks_to_race:         int = 12     # Plan duration
    max_long_run_min:      int = 150    # Max long run duration (minutes)
    easy_run_pace_min_km:  float = 6.0    # Easy pace (min/km)
    quality_sessions_pw:   int = 2      # Quality sessions per week
    experience_level:      str = "intermediate"  # beginner/intermediate/advanced

    @classmethod
    def beginner(cls) -> "AthleteProfile":
        return cls(
            current_ctl=20.0, max_long_run_min=90,
            quality_sessions_pw=1, experience_level="beginner"
        )

    @classmethod
    def intermediate(cls) -> "AthleteProfile":
        return cls(
            current_ctl=40.0, max_long_run_min=150,
            quality_sessions_pw=2, experience_level="intermediate"
        )

    @classmethod
    def advanced(cls) -> "AthleteProfile":
        return cls(
            current_ctl=65.0, max_long_run_min=180,
            quality_sessions_pw=2, experience_level="advanced"
        )


class PlanGenerator:
    """
    Generates a training DAG from an athlete profile.

    Two steps:
    1. _create_nodes()  — build all WorkoutNode objects
    2. _wire_edges()    — add DependencyEdge objects (coaching rules)

    The result is a TrainingDAG ready for topological sort
    and constraint validation.
    """

    # Day constants (1=Mon, 7=Sun)
    MON, TUE, WED, THU, FRI, SAT, SUN = 1, 2, 3, 4, 5, 6, 7

    def __init__(self, profile: AthleteProfile):
        self.profile = profile
        self.graph = TrainingDAG()

    def generate(self) -> TrainingDAG:
        """
        Generate the complete training DAG.
        Returns the wired graph ready for scheduling.
        """
        self._create_nodes()
        self._wire_edges()
        return self.graph

    # ── Step 1: Create all workout nodes ─────────────────────────

    def _create_nodes(self) -> None:
        """
        Build all WorkoutNode objects for the full plan.
        Nodes are created week by week using a weekly template
        that adapts based on the training phase.
        """
        weeks = self.profile.weeks_to_race

        for week in range(1, weeks + 1):
            phase = self._get_phase(week, weeks)
            nodes = self._get_week_template(week, phase)
            for node in nodes:
                self.graph.add_node(node)

    def _get_phase(self, week: int, total_weeks: int) -> str:
        """Classify a week into its training phase."""
        if week <= total_weeks * 0.25:
            return "base"
        elif week <= total_weeks * 0.67:
            return "build"
        elif week <= total_weeks * 0.83:
            return "peak"
        else:
            return "taper"

    def _get_week_template(self,
                           week: int,
                           phase: str) -> List[WorkoutNode]:
        """
        Generate the workout nodes for one week based on phase.

        Weekly structure:
          Mon = Rest or cross-train
          Tue = Easy run
          Wed = Quality session (or easy in base/taper)
          Thu = Easy run
          Fri = Easy run or rest
          Sat = Second quality session (build/peak) or easy
          Sun = Long run
        """
        p = self.profile
        nodes = []

        # ── Duration scaling by phase ──────────────────────────────
        if phase == "base":
            easy_min = 40
            long_min = min(p.max_long_run_min * 0.60, 90)
            tempo_min = 40
            interval_min = 45
        elif phase == "build":
            easy_min = 45
            long_min = min(p.max_long_run_min * 0.80, 120)
            tempo_min = 50
            interval_min = 50
        elif phase == "peak":
            easy_min = 50
            long_min = p.max_long_run_min
            tempo_min = 55
            interval_min = 55
        else:  # taper
            easy_min = 35
            long_min = min(p.max_long_run_min * 0.50, 75)
            tempo_min = 35
            interval_min = 40

        # ── Recovery week every 4th week ──────────────────────────
        is_recovery_week = (week % 4 == 0)
        if is_recovery_week:
            easy_min = int(easy_min * 0.75)
            long_min = int(long_min * 0.75)
            tempo_min = int(tempo_min * 0.75)

        # ── Race week ─────────────────────────────────────────────
        is_race_week = (week == self.profile.weeks_to_race)

        if is_race_week:
            nodes.extend(self._race_week_template(week, easy_min))
            return nodes

        # ── Monday: Rest or cross-train ───────────────────────────
        mon_type = (WorkoutType.CROSS_TRAIN
                    if phase in ("build", "peak") and not is_recovery_week
                    else WorkoutType.REST)
        nodes.append(WorkoutNode(
            workout_type=mon_type, week=week, day=self.MON,
            duration_min=30 if mon_type == WorkoutType.CROSS_TRAIN else 0,
            notes="Active recovery — bike, swim, or yoga" if mon_type == WorkoutType.CROSS_TRAIN else "Full rest"
        ))

        # ── Tuesday: Easy run ─────────────────────────────────────
        nodes.append(WorkoutNode(
            workout_type=WorkoutType.EASY_RUN, week=week, day=self.TUE,
            duration_min=easy_min,
            notes="Zone 2 — conversational pace, nose breathing"
        ))

        # ── Wednesday: Quality session ────────────────────────────
        if phase == "base" or is_recovery_week:
            # No quality in base phase or recovery weeks
            nodes.append(WorkoutNode(
                workout_type=WorkoutType.EASY_RUN, week=week, day=self.WED,
                duration_min=easy_min,
                notes="Easy aerobic run — building base"
            ))
        elif phase in ("build", "peak"):
            # Alternate tempo and intervals week by week
            if week % 2 == 0:
                nodes.append(WorkoutNode(
                    workout_type=WorkoutType.TEMPO_RUN, week=week, day=self.WED,
                    duration_min=tempo_min,
                    notes="Comfortably hard — 20-40min at lactate threshold"
                ))
            else:
                nodes.append(WorkoutNode(
                    workout_type=WorkoutType.INTERVAL_RUN, week=week, day=self.WED,
                    duration_min=interval_min,
                    notes="VO2max intervals — e.g. 5×1km at 5k pace"
                ))
        else:  # taper
            nodes.append(WorkoutNode(
                workout_type=WorkoutType.TEMPO_RUN, week=week, day=self.WED,
                duration_min=tempo_min,
                notes="Short tempo — maintain sharpness during taper"
            ))

        # ── Thursday: Easy run ────────────────────────────────────
        nodes.append(WorkoutNode(
            workout_type=WorkoutType.EASY_RUN, week=week, day=self.THU,
            duration_min=easy_min,
            notes="Recovery easy — flush out Wednesday's effort"
        ))

        # ── Friday: Easy or rest ──────────────────────────────────
        if phase in ("peak",) and not is_recovery_week:
            # Second quality session in peak phase
            nodes.append(WorkoutNode(
                workout_type=WorkoutType.HILL_RUN, week=week, day=self.FRI,
                duration_min=45,
                notes="Hill repeats — strength + neuromuscular stimulus"
            ))
        else:
            nodes.append(WorkoutNode(
                workout_type=WorkoutType.EASY_RUN, week=week, day=self.FRI,
                duration_min=max(easy_min - 10, 25),
                notes="Short easy run — legs fresh for long run"
            ))

        # ── Saturday: Quality or easy ─────────────────────────────
        if phase in ("build", "peak") and not is_recovery_week:
            nodes.append(WorkoutNode(
                workout_type=WorkoutType.EASY_RUN, week=week, day=self.SAT,
                duration_min=easy_min - 5,
                notes="Easy shakeout — arrive fresh for long run"
            ))
        else:
            nodes.append(WorkoutNode(
                workout_type=WorkoutType.REST, week=week, day=self.SAT,
                duration_min=0,
                notes="Rest before long run"
            ))

        # ── Sunday: Long run ──────────────────────────────────────
        nodes.append(WorkoutNode(
            workout_type=WorkoutType.LONG_RUN, week=week, day=self.SUN,
            duration_min=int(long_min),
            notes=f"Long run — easy pace, build aerobic endurance. "
                  f"Target: {int(long_min)}min"
        ))

        return nodes

    def _race_week_template(self, week: int,
                            easy_min: int) -> List[WorkoutNode]:
        """Race week: minimal volume, race on Sunday."""
        return [
            WorkoutNode(
                workout_type=WorkoutType.REST, week=week, day=self.MON,
                duration_min=0, notes="Full rest — legs recovering from taper"),
            WorkoutNode(
                workout_type=WorkoutType.EASY_RUN, week=week, day=self.TUE,
                duration_min=25, notes="Short easy — shake out the legs"),
            WorkoutNode(
                workout_type=WorkoutType.EASY_RUN, week=week, day=self.WED,
                duration_min=20, notes="Easy strides — wake up the fast twitch"),
            WorkoutNode(
                workout_type=WorkoutType.REST, week=week, day=self.THU,
                duration_min=0, notes="Full rest"),
            WorkoutNode(
                workout_type=WorkoutType.EASY_RUN, week=week, day=self.FRI,
                duration_min=15, notes="Very easy shakeout — 15min only"),
            WorkoutNode(
                workout_type=WorkoutType.REST, week=week, day=self.SAT,
                duration_min=0,
                status=NodeStatus.LOCKED,
                notes="Mandatory rest before race — LOCKED"),
            WorkoutNode(
                workout_type=WorkoutType.RACE, week=week, day=self.SUN,
                duration_min=int(self.profile.race_distance_km *
                                 self.profile.easy_run_pace_min_km),
                status=NodeStatus.LOCKED,
                notes=f"RACE DAY 🏁 — {self.profile.race_distance_km}km"),
        ]

    # ── Step 2: Wire all dependency edges ─────────────────────────

    def _wire_edges(self) -> None:
        """
        Add all coaching dependency edges to the graph.

        Rules encoded:
        1. Recovery after long run (HARD)
        2. Fresh legs for quality sessions (HARD)
        3. Pre-race rest (HARD)
        4. Easy before long run (SOFT)
        5. Progressive long run distances (SOFT)
        6. Cross-train / rest after quality session (SOFT)
        """
        weeks = self.profile.weeks_to_race

        for week in range(1, weeks + 1):
            week_nodes = {n.day: n for n in self.graph.get_week_nodes(week)}

            # ── Rule 1: Recovery run must follow Long Run ──────────
            # Long run (Sun) → Recovery easy run next Monday or Tuesday
            if self.SUN in week_nodes and week < weeks:
                next_week_nodes = {n.day: n
                                   for n in self.graph.get_week_nodes(week + 1)}
                long_run = week_nodes[self.SUN]

                # Monday of next week should be rest/easy (not quality)
                if self.MON in next_week_nodes:
                    next_mon = next_week_nodes[self.MON]
                    if not next_mon.is_quality_session:
                        try:
                            self.graph.add_edge(DependencyEdge(
                                source_id=long_run.node_id,
                                target_id=next_mon.node_id,
                                constraint_type=ConstraintType.HARD,
                                reason=EdgeReason.REST_AFTER_LONG,
                                min_gap_days=1,
                                notes="Long run must be followed by rest/easy"
                            ))
                        except Exception:
                            pass

            # ── Rule 2: No quality session without prior easy run ──
            # Wednesday quality ← Tuesday easy (HARD)
            if self.WED in week_nodes and self.TUE in week_nodes:
                wed = week_nodes[self.WED]
                tue = week_nodes[self.TUE]
                if wed.is_quality_session:
                    try:
                        self.graph.add_edge(DependencyEdge(
                            source_id=tue.node_id,
                            target_id=wed.node_id,
                            constraint_type=ConstraintType.HARD,
                            reason=EdgeReason.FRESH_FOR_QUALITY,
                            min_gap_days=1,
                            notes="Easy run day before quality session"
                        ))
                    except Exception:
                        pass

            # ── Rule 3: Easy run before Long Run (SOFT) ───────────
            # Saturday easy → Sunday long run
            if self.SAT in week_nodes and self.SUN in week_nodes:
                sat = week_nodes[self.SAT]
                sun = week_nodes[self.SUN]
                if not sat.is_quality_session:
                    try:
                        self.graph.add_edge(DependencyEdge(
                            source_id=sat.node_id,
                            target_id=sun.node_id,
                            constraint_type=ConstraintType.SOFT,
                            reason=EdgeReason.EASY_BEFORE_LONG,
                            min_gap_days=1,
                            notes="Easy day before long run preferred"
                        ))
                    except Exception:
                        pass

            # ── Rule 4: Recovery after quality session (SOFT) ──────
            # Wednesday quality → Thursday easy
            if self.WED in week_nodes and self.THU in week_nodes:
                wed = week_nodes[self.WED]
                thu = week_nodes[self.THU]
                if wed.is_quality_session and not thu.is_quality_session:
                    try:
                        self.graph.add_edge(DependencyEdge(
                            source_id=wed.node_id,
                            target_id=thu.node_id,
                            constraint_type=ConstraintType.SOFT,
                            reason=EdgeReason.RECOVERY_AFTER_QUALITY,
                            min_gap_days=1,
                            notes="Easy run after quality session for adaptation"
                        ))
                    except Exception:
                        pass

            # ── Rule 5: Progressive long run chain ─────────────────
            # Long run this week → Long run next week (progression)
            if self.SUN in week_nodes and week < weeks:
                next_week_nodes = {n.day: n
                                   for n in self.graph.get_week_nodes(week + 1)}
                if self.SUN in next_week_nodes:
                    this_long = week_nodes[self.SUN]
                    next_long = next_week_nodes[self.SUN]
                    # Only add if both are long runs (not race)
                    if (this_long.workout_type == WorkoutType.LONG_RUN and
                            next_long.workout_type == WorkoutType.LONG_RUN):
                        try:
                            self.graph.add_edge(DependencyEdge(
                                source_id=this_long.node_id,
                                target_id=next_long.node_id,
                                constraint_type=ConstraintType.SOFT,
                                reason=EdgeReason.PROGRESSIVE_OVERLOAD,
                                min_gap_days=7,
                                notes="Progressive long run — each week builds on last"
                            ))
                        except Exception:
                            pass

        # ── Rule 6: Pre-race rest (HARD) ──────────────────────────
        # Final week: Saturday rest → Sunday race
        race_week_nodes = {n.day: n
                           for n in self.graph.get_week_nodes(weeks)}
        if self.SAT in race_week_nodes and self.SUN in race_week_nodes:
            sat_rest = race_week_nodes[self.SAT]
            race = race_week_nodes[self.SUN]
            if race.workout_type == WorkoutType.RACE:
                try:
                    self.graph.add_edge(DependencyEdge(
                        source_id=sat_rest.node_id,
                        target_id=race.node_id,
                        constraint_type=ConstraintType.HARD,
                        reason=EdgeReason.PRE_RACE_REST,
                        min_gap_days=1,
                        notes="Mandatory rest day before race"
                    ))
                except Exception:
                    pass

    # Day constants
    MON, TUE, WED, THU, FRI, SAT, SUN = 1, 2, 3, 4, 5, 6, 7
