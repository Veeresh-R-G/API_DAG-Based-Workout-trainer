"""
replanner.py — DynamicReplanner
--------------------------------
Handles missed workouts and dynamically recomputes the plan.

When an athlete misses a workout, several things can happen:
1. The missed workout can be shifted to the next available day
2. Downstream workouts dependent on the missed one need rescheduling
3. If rescheduling isn't possible, the workout is dropped (SKIPPED)
4. The weekly load budget may need rebalancing

This is a graph reachability problem:
- Find all descendants of the missed node (BFS)
- Determine which have unmet hard prerequisites
- Try to reschedule them
- If not possible, mark SKIPPED and propagate

The replanner returns a ReplanningSummary describing what changed
and why — this is displayed in the UI so the athlete understands
the impact of their missed workout.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum

from core.nodes import WorkoutNode, WorkoutType, NodeStatus
from core.edges import ConstraintType
from core.graph import TrainingDAG


class ReplannedAction(Enum):
    COMPLETED = "completed"    # Marked as done
    MISSED = "missed"       # Marked as missed, no reschedule possible
    RESCHEDULED = "rescheduled"  # Moved to a different day
    SKIPPED = "skipped"      # Dropped from plan (downstream of missed)
    NO_CHANGE = "no_change"    # Not affected


@dataclass
class NodeChange:
    """Describes what happened to a single node during replanning."""
    node_id:     str
    action:      ReplannedAction
    reason:      str
    old_week:    Optional[int] = None
    old_day:     Optional[int] = None
    new_week:    Optional[int] = None
    new_day:     Optional[int] = None


@dataclass
class ReplanningSummary:
    """
    Full summary of what the replanner did.
    Returned to the frontend for display.
    """
    triggered_by:     str                   # node_id that was missed
    changes:          List[NodeChange] = field(default_factory=list)
    warnings:         List[str] = field(default_factory=list)
    affected_count:   int = 0
    plan_quality_before: float = 0.0
    plan_quality_after:  float = 0.0

    def add_change(self, change: NodeChange) -> None:
        self.changes.append(change)
        self.affected_count = len(self.changes)

    def to_dict(self) -> dict:
        return {
            "triggered_by":        self.triggered_by,
            "affected_count":      self.affected_count,
            "plan_quality_before": self.plan_quality_before,
            "plan_quality_after":  self.plan_quality_after,
            "warnings":            self.warnings,
            "changes": [
                {
                    "node_id":  c.node_id,
                    "action":   c.action.value,
                    "reason":   c.reason,
                    "old_week": c.old_week,
                    "old_day":  c.old_day,
                    "new_week": c.new_week,
                    "new_day":  c.new_day,
                }
                for c in self.changes
            ],
        }


class DynamicReplanner:
    """
    Handles missed workouts and dynamically replans the DAG.

    Strategy:
    1. Mark the node as MISSED
    2. Find all affected descendants (BFS)
    3. For each affected node, check if its hard prerequisites
       are still satisfiable
    4. Try to shift the affected node to the next available slot
    5. If shifting fails, mark SKIPPED and continue propagation
    6. Revalidate the plan and report quality score change
    """

    def __init__(self, graph: TrainingDAG):
        self.graph = graph

    def mark_complete(self, node_id: str) -> ReplanningSummary:
        node = self.graph.get_node(node_id)
        quality_before = self.graph.get_plan_quality_score()
        node.mark_complete()

        summary = ReplanningSummary(
            triggered_by=node_id,
            plan_quality_before=quality_before,
        )
        summary.add_change(NodeChange(
            node_id=node_id,
            action=ReplannedAction.COMPLETED,
            reason=f"Athlete completed {node.label}",
        ))

        # ── NEW: Check what this completion unlocks ────────────────
        # Find successors whose ALL hard prerequisites are now complete
        newly_unlocked = []
        for succ in self.graph.get_successors(node_id):
            hard_preds = [
                e for e in self.graph.get_edges_to(succ.node_id)
                if e.is_hard
            ]
            all_satisfied = all(
                self.graph.get_node(e.source_id).status
                in (NodeStatus.COMPLETE, NodeStatus.SKIPPED)
                for e in hard_preds
            )
            if all_satisfied and succ.status == NodeStatus.PENDING:
                newly_unlocked.append(succ)
                summary.add_change(NodeChange(
                    node_id=succ.node_id,
                    action=ReplannedAction.NO_CHANGE,
                    reason=f"{succ.label} is now unlocked — all prerequisites complete",
                ))

        summary.plan_quality_after = self.graph.get_plan_quality_score()
        return summary

    def mark_missed(self, node_id: str) -> ReplanningSummary:
        """
        Mark a workout as missed and trigger replanning.

        1. Mark the node MISSED
        2. Find all hard-constraint descendants
        3. Try to reschedule or skip them
        4. Return summary of all changes
        """
        node = self.graph.get_node(node_id)
        quality_before = self.graph.get_plan_quality_score()
        node.mark_missed()

        summary = ReplanningSummary(
            triggered_by=node_id,
            plan_quality_before=quality_before,
        )
        summary.add_change(NodeChange(
            node_id=node_id,
            action=ReplannedAction.MISSED,
            reason=f"Athlete missed {node.label}",
            old_week=node.week,
            old_day=node.day,
        ))

        # Find all descendants with hard dependencies on missed node
        affected_ids = self._find_hard_affected(node_id)

        for affected_id in affected_ids:
            affected = self.graph.get_node(affected_id)

            # Skip locked nodes (race day, etc.)
            if affected.status == NodeStatus.LOCKED:
                summary.warnings.append(
                    f"⚠️  {affected.label} (W{affected.week}D{affected.day}) "
                    f"is LOCKED and cannot be replanned")
                continue

            # Check if hard prerequisites are still satisfied
            unmet = self._get_unmet_hard_prereqs(affected_id)

            if not unmet:
                # Prerequisites still met — no change needed
                continue

            # Try to reschedule
            new_slot = self._find_next_available_slot(affected)
            if new_slot:
                new_week, new_day = new_slot
                summary.add_change(NodeChange(
                    node_id=affected_id,
                    action=ReplannedAction.RESCHEDULED,
                    reason=(f"{affected.label} rescheduled: "
                            f"prerequisite {node.label} was missed"),
                    old_week=affected.week,
                    old_day=affected.day,
                    new_week=new_week,
                    new_day=new_day,
                ))
                # Note: actual rescheduling would require mutable week/day
                # For now we flag it — full implementation in Phase 2
            else:
                # Can't reschedule — skip this workout
                affected.status = NodeStatus.SKIPPED
                summary.add_change(NodeChange(
                    node_id=affected_id,
                    action=ReplannedAction.SKIPPED,
                    reason=(f"{affected.label} dropped: "
                            f"cannot reschedule after missed {node.label}"),
                    old_week=affected.week,
                    old_day=affected.day,
                ))

                if affected.is_quality_session:
                    summary.warnings.append(
                        f"⚠️  Quality session {affected.label} "
                        f"(W{affected.week}D{affected.day}) was dropped. "
                        f"This may affect fitness development.")

        summary.plan_quality_after = self.graph.get_plan_quality_score()
        return summary

    # ── Private helpers ───────────────────────────────────────────

    def _find_hard_affected(self, missed_node_id: str) -> List[str]:
        """
        BFS through HARD edges to find all transitively affected nodes.

        Example:
        A ──HARD──→ B ──HARD──→ C ──HARD──→ D

        Miss A → affects B, C, D (not just B)
        """
        affected = []
        visited = {missed_node_id}
        queue = [missed_node_id]

        while queue:
            current_id = queue.pop(0)
            for edge in self.graph.get_edges_from(current_id):
                if edge.is_hard and edge.target_id not in visited:
                    visited.add(edge.target_id)
                    affected.append(edge.target_id)
                    queue.append(edge.target_id)   # ← key: recurse

        return affected

    def _get_unmet_hard_prereqs(self, node_id: str) -> List[str]:
        """
        Find hard prerequisite nodes that are MISSED or SKIPPED.
        These make this node's scheduling constraint unmet.
        """
        unmet = []
        hard_edges_in = [
            e for e in self.graph.get_edges_to(node_id)
            if e.is_hard
        ]
        for edge in hard_edges_in:
            src = self.graph.get_node(edge.source_id)
            if src.status in (NodeStatus.MISSED, NodeStatus.SKIPPED):
                unmet.append(edge.source_id)
        return unmet

    def _find_next_available_slot(
        self, node: WorkoutNode
    ) -> Optional[tuple]:
        """
        Find the next available (week, day) slot after the node's
        current position that doesn't conflict with existing workouts.

        Returns (week, day) or None if no slot found within plan.
        """
        total_weeks = self.graph.get_week_count()

        # Try the next 3 days
        current_abs = (node.week - 1) * 7 + node.day
        for delta in range(1, 4):
            abs_day = current_abs + delta
            new_week = (abs_day - 1) // 7 + 1
            new_day = (abs_day - 1) % 7 + 1

            if new_week > total_weeks:
                return None

            # Check if slot is free (no workout already there)
            existing = [
                n for n in self.graph.get_week_nodes(new_week)
                if n.day == new_day and n.status != NodeStatus.SKIPPED
            ]
            if not existing:
                return (new_week, new_day)

        return None
