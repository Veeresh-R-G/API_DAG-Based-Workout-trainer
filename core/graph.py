"""
graph.py — TrainingDAG
-----------------------
The core data structure of the training planner.

A directed acyclic graph where:
  Nodes = WorkoutNode instances
  Edges = DependencyEdge instances (coaching rules)

Key operations:
  add_node()        → add a workout to the plan
  add_edge()        → add a coaching dependency
  get_predecessors()→ what must happen before this workout
  get_successors()  → what does this workout unlock
  has_cycle()       → validate the graph is acyclic
  topological_sort()→ Kahn's algorithm — valid execution order
  get_week_nodes()  → all workouts in a given week

Why DAG (not a general graph)?
A cycle would mean "A must come before B AND B must come before A"
which is logically impossible for scheduling. The acyclic property
guarantees a valid topological ordering always exists.

Topological sort — Kahn's Algorithm (1962):
  1. Find all nodes with in-degree 0 (no unmet prerequisites)
  2. Add them to the schedule
  3. Remove their edges, update in-degrees
  4. Repeat until all nodes are scheduled
  5. If nodes remain but none have in-degree 0 → cycle detected

Time complexity: O(V + E) where V = workouts, E = dependencies
This is worth mentioning in interviews — optimal for this problem.
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque

from .nodes import WorkoutNode, WorkoutType, NodeStatus
from .edges import DependencyEdge, ConstraintType


class CycleDetectedError(Exception):
    """Raised when a cycle is detected in the training DAG."""
    pass


class NodeNotFoundError(Exception):
    """Raised when a referenced node_id doesn't exist in the graph."""
    pass


class TrainingDAG:
    """
    Directed Acyclic Graph representing a running training plan.

    The graph stores workout nodes and dependency edges.
    All scheduling logic, constraint checking, and replanning
    operates on this graph.

    Internal representation:
        _nodes:      {node_id: WorkoutNode}
        _edges:      [DependencyEdge]
        _adj:        {node_id: set of successor node_ids}  (forward)
        _radj:       {node_id: set of predecessor node_ids} (reverse)
    """

    def __init__(self, plan_id: Optional[str] = None):
        import uuid
        self.plan_id:  str = plan_id or str(uuid.uuid4())[:8]
        self._nodes:   Dict[str, WorkoutNode] = {}
        self._edges:   List[DependencyEdge]   = []
        self._adj:     Dict[str, Set[str]]    = defaultdict(set)
        self._radj:    Dict[str, Set[str]]    = defaultdict(set)

    # ── Node operations ───────────────────────────────────────────

    def add_node(self, node: WorkoutNode) -> None:
        """Add a workout node to the graph."""
        if node.node_id in self._nodes:
            raise ValueError(f"Node {node.node_id} already exists")
        self._nodes[node.node_id] = node
        # Ensure adjacency entries exist even for isolated nodes
        if node.node_id not in self._adj:
            self._adj[node.node_id] = set()
        if node.node_id not in self._radj:
            self._radj[node.node_id] = set()

    def get_node(self, node_id: str) -> WorkoutNode:
        """Get a node by ID."""
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node {node_id} not found")
        return self._nodes[node_id]

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node {node_id} not found")
        # Remove all edges involving this node
        self._edges = [e for e in self._edges
                       if e.source_id != node_id
                       and e.target_id != node_id]
        # Update adjacency
        for succ in list(self._adj.get(node_id, [])):
            self._radj[succ].discard(node_id)
        for pred in list(self._radj.get(node_id, [])):
            self._adj[pred].discard(node_id)
        del self._adj[node_id]
        del self._radj[node_id]
        del self._nodes[node_id]

    @property
    def nodes(self) -> List[WorkoutNode]:
        """All workout nodes, sorted by week then day."""
        return sorted(self._nodes.values(),
                      key=lambda n: (n.week, n.day))

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    # ── Edge operations ───────────────────────────────────────────

    def add_edge(self, edge: DependencyEdge) -> None:
        """
        Add a dependency edge to the graph.
        Validates both endpoints exist and checks for cycles.
        """
        if edge.source_id not in self._nodes:
            raise NodeNotFoundError(f"Source node {edge.source_id} not found")
        if edge.target_id not in self._nodes:
            raise NodeNotFoundError(f"Target node {edge.target_id} not found")
        if edge.source_id == edge.target_id:
            raise ValueError("Self-loops are not allowed")

        # Check cycle before adding
        if self._would_create_cycle(edge.source_id, edge.target_id):
            raise CycleDetectedError(
                f"Adding edge {edge.source_id} → {edge.target_id} "
                f"would create a cycle")

        self._edges.append(edge)
        self._adj[edge.source_id].add(edge.target_id)
        self._radj[edge.target_id].add(edge.source_id)

    def get_edges_from(self, node_id: str) -> List[DependencyEdge]:
        """All edges where this node is the source."""
        return [e for e in self._edges if e.source_id == node_id]

    def get_edges_to(self, node_id: str) -> List[DependencyEdge]:
        """All edges where this node is the target."""
        return [e for e in self._edges if e.target_id == node_id]

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    # ── Graph traversal ───────────────────────────────────────────

    def get_predecessors(self, node_id: str) -> List[WorkoutNode]:
        """
        All nodes that must be completed before this node.
        (Direct predecessors only — not transitive)
        """
        return [self._nodes[nid]
                for nid in self._radj.get(node_id, set())
                if nid in self._nodes]

    def get_successors(self, node_id: str) -> List[WorkoutNode]:
        """
        All nodes that this node unlocks.
        (Direct successors only — not transitive)
        """
        return [self._nodes[nid]
                for nid in self._adj.get(node_id, set())
                if nid in self._nodes]

    def get_all_ancestors(self, node_id: str) -> Set[str]:
        """
        All transitive predecessors (BFS on reverse graph).
        Used to find everything that must happen before a node.
        """
        visited = set()
        queue   = deque([node_id])
        while queue:
            current = queue.popleft()
            for pred_id in self._radj.get(current, set()):
                if pred_id not in visited:
                    visited.add(pred_id)
                    queue.append(pred_id)
        return visited

    def get_all_descendants(self, node_id: str) -> Set[str]:
        """
        All transitive successors (BFS on forward graph).
        Used to find everything a missed workout affects.
        """
        visited = set()
        queue   = deque([node_id])
        while queue:
            current = queue.popleft()
            for succ_id in self._adj.get(current, set()):
                if succ_id not in visited:
                    visited.add(succ_id)
                    queue.append(succ_id)
        return visited

    # ── Topological sort (Kahn's Algorithm) ───────────────────────

    def topological_sort(self) -> List[WorkoutNode]:
        """
        Kahn's Algorithm (1962) — O(V + E).

        Returns all workout nodes in a valid execution order:
        if edge A → B exists, A appears before B in the result.

        This is the schedule. The order is the plan.

        Algorithm:
        1. Compute in-degree for all nodes
        2. Enqueue nodes with in-degree 0 (no prerequisites)
        3. Process: dequeue node, add to result, decrement
           in-degrees of successors, enqueue any that reach 0
        4. If result length < node count → cycle exists

        Note: Multiple valid orderings may exist.
        We use (week, day) as a secondary sort key to produce
        a schedule that respects the intended weekly structure.
        """
        # Compute in-degrees
        in_degree = {nid: 0 for nid in self._nodes}
        for nid in self._nodes:
            for succ_id in self._adj.get(nid, set()):
                in_degree[succ_id] = in_degree.get(succ_id, 0) + 1

        # Initialise queue with nodes that have no prerequisites
        # Sort by (week, day) for deterministic, calendar-ordered output
        queue = deque(sorted(
            [nid for nid, deg in in_degree.items() if deg == 0],
            key=lambda nid: (self._nodes[nid].week, self._nodes[nid].day)
        ))

        result = []
        while queue:
            node_id = queue.popleft()
            result.append(self._nodes[node_id])

            # Remove this node's edges, update in-degrees
            successors = sorted(
                list(self._adj.get(node_id, set())),
                key=lambda nid: (self._nodes[nid].week, self._nodes[nid].day)
            )
            for succ_id in successors:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)

        # Cycle check
        if len(result) != len(self._nodes):
            raise CycleDetectedError(
                f"Cycle detected: only {len(result)} of "
                f"{len(self._nodes)} nodes were sorted")

        return result

    def has_cycle(self) -> bool:
        """Check if the graph contains a cycle."""
        try:
            self.topological_sort()
            return False
        except CycleDetectedError:
            return True

    # ── Week-level queries ────────────────────────────────────────

    def get_week_nodes(self, week: int) -> List[WorkoutNode]:
        """All workout nodes in a given week, sorted by day."""
        return sorted(
            [n for n in self._nodes.values() if n.week == week],
            key=lambda n: n.day
        )

    def get_week_trimp(self, week: int) -> float:
        """Total TRIMP load for a given week."""
        return sum(n.trimp for n in self.get_week_nodes(week)
                   if n.status != NodeStatus.MISSED)

    def get_week_count(self) -> int:
        """Total number of weeks in the plan."""
        if not self._nodes:
            return 0
        return max(n.week for n in self._nodes.values())

    def get_quality_sessions_in_week(self, week: int) -> List[WorkoutNode]:
        """All quality (hard) sessions in a given week."""
        return [n for n in self.get_week_nodes(week)
                if n.is_quality_session]

    # ── Constraint validation ─────────────────────────────────────

    def validate_hard_constraints(self) -> List[str]:
        """
        Check all HARD constraint edges are satisfiable.
        Returns list of violation descriptions (empty = valid).
        """
        violations = []
        hard_edges = [e for e in self._edges if e.is_hard]

        for edge in hard_edges:
            src  = self._nodes[edge.source_id]
            tgt  = self._nodes[edge.target_id]

            # Source must come before target chronologically
            src_abs_day = (src.week - 1) * 7 + src.day
            tgt_abs_day = (tgt.week - 1) * 7 + tgt.day
            gap = tgt_abs_day - src_abs_day

            if gap < edge.min_gap_days:
                violations.append(
                    f"HARD violation: {src.label} (W{src.week}D{src.day}) → "
                    f"{tgt.label} (W{tgt.week}D{tgt.day}): "
                    f"gap={gap}d, required≥{edge.min_gap_days}d. "
                    f"Reason: {edge.description}"
                )
            if edge.max_gap_days and gap > edge.max_gap_days:
                violations.append(
                    f"HARD violation: {src.label} (W{src.week}D{src.day}) → "
                    f"{tgt.label} (W{tgt.week}D{tgt.day}): "
                    f"gap={gap}d, max={edge.max_gap_days}d. "
                    f"Reason: {edge.description}"
                )

        return violations

    def get_plan_quality_score(self) -> float:
        """
        Score from 0-100 based on how many soft constraints are met.
        Used to compare alternative plan configurations.
        """
        soft_edges = [e for e in self._edges if e.is_soft]
        if not soft_edges:
            return 100.0

        satisfied = 0
        for edge in soft_edges:
            src = self._nodes[edge.source_id]
            tgt = self._nodes[edge.target_id]
            src_abs = (src.week - 1) * 7 + src.day
            tgt_abs = (tgt.week - 1) * 7 + tgt.day
            gap = tgt_abs - src_abs
            if gap >= edge.min_gap_days:
                satisfied += 1

        return round(satisfied / len(soft_edges) * 100, 1)

    # ── Serialisation ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Serialise the full graph to JSON-compatible dict.
        This is what the FastAPI endpoint returns to the frontend.
        ReactFlow consumes 'nodes' and 'edges' directly.
        """
        topo_order = self.topological_sort()

        return {
            "plan_id":       self.plan_id,
            "node_count":    self.node_count,
            "edge_count":    self.edge_count,
            "week_count":    self.get_week_count(),
            "quality_score": self.get_plan_quality_score(),
            "nodes":         [n.to_dict() for n in topo_order],
            "edges":         [e.to_dict() for e in self._edges],
            "weeks": {
                str(w): {
                    "week":      w,
                    "trimp":     round(self.get_week_trimp(w), 1),
                    "workouts":  [n.to_dict() for n in self.get_week_nodes(w)],
                    "quality_sessions": len(self.get_quality_sessions_in_week(w)),
                }
                for w in range(1, self.get_week_count() + 1)
            },
        }

    # ── Internal helpers ──────────────────────────────────────────

    def _would_create_cycle(self, src_id: str, tgt_id: str) -> bool:
        """
        Check if adding edge src→tgt would create a cycle.
        Uses DFS from tgt to check if src is reachable from tgt.
        If yes, adding src→tgt would close the cycle.
        """
        visited = set()
        stack   = [tgt_id]
        while stack:
            current = stack.pop()
            if current == src_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self._adj.get(current, set()))
        return False

    def __repr__(self) -> str:
        return (f"TrainingDAG(plan_id={self.plan_id}, "
                f"nodes={self.node_count}, "
                f"edges={self.edge_count}, "
                f"weeks={self.get_week_count()})")

    # ── Private cycle detection ───────────────────────────────────

    def _would_create_cycle(self, src_id: str, tgt_id: str) -> bool:
        """
        DFS check: is src_id reachable from tgt_id?
        If yes, adding src→tgt closes a cycle.
        """
        visited = set()
        stack   = [tgt_id]
        while stack:
            node = stack.pop()
            if node == src_id:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self._adj.get(node, []))
        return False
