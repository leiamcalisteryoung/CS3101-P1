from dataclasses import dataclass
from itertools import permutations

from builder import ProgramState
from query_models import (
    AndPredicate,
    AttrEqAttrPredicate,
    AttrEqConstPredicate,
    DifferenceQuery,
    EmptyQuery,
    JoinQuery,
    Predicate,
    ProjectQuery,
    QueryExpr,
    RelVarQuery,
    RenameQuery,
    SelectQuery,
    UnionQuery,
    format_query_expr,
)


class CostBasedJoinOptimizer:
    """Phase 2 optimizer: cost-based optimization of join chains.

    This class focuses only on join-chain reordering decisions.
    It runs as a separate phase after heuristic rewrites are finished.
    """

    def __init__(self, state: ProgramState) -> None:
        self._state = state

    # Public entry for phase 2.
    # Runs one bottom-up sweep and returns (rewritten_expr, phase2_steps)
    # where phase2_steps is a list of "Rule: expression" lines.
    def optimize(self, expr: QueryExpr) -> tuple[QueryExpr, list[str]]:
        if isinstance(expr, SelectQuery):
            new_source, steps = self.optimize(expr.source)
            return SelectQuery(source=new_source, predicate=expr.predicate), steps

        if isinstance(expr, ProjectQuery):
            new_source, steps = self.optimize(expr.source)
            return ProjectQuery(source=new_source, attributes=list(expr.attributes)), steps

        if isinstance(expr, RenameQuery):
            new_source, steps = self.optimize(expr.source)
            return RenameQuery(source=new_source, new_attributes=list(expr.new_attributes)), steps

        if isinstance(expr, UnionQuery):
            new_left, left_steps = self.optimize(expr.left)
            new_right, right_steps = self.optimize(expr.right)
            return UnionQuery(left=new_left, right=new_right), (left_steps + right_steps)

        if isinstance(expr, DifferenceQuery):
            new_left, left_steps = self.optimize(expr.left)
            new_right, right_steps = self.optimize(expr.right)
            return DifferenceQuery(left=new_left, right=new_right), (left_steps + right_steps)

        if isinstance(expr, JoinQuery):
            # 1) Optimize deeper joins first.
            new_left, left_steps = self.optimize(expr.left)
            new_right, right_steps = self.optimize(expr.right)

            candidate = JoinQuery(left=new_left, right=new_right)
            steps = left_steps + right_steps

            # 2) Then optimize this join node using local cost-based alternatives.
            rewritten, rule_name = self._apply_cost_based_join_reorder(candidate)
            if rewritten is not None:
                steps.append(f"{rule_name}: {format_query_expr(rewritten)}")
                return rewritten, steps

            return candidate, steps

        # Leaf nodes (RelVarQuery / EmptyQuery) have no join-chain work.
        return expr, []

    # Decide local join reordering via cost comparison.
    def _apply_cost_based_join_reorder(self, expr: JoinQuery) -> tuple[QueryExpr | None, str | None]:
        # Get the join frontier operands for this node, if it matches a recognized shape.
        left_is_join = isinstance(expr.left, JoinQuery)
        right_is_join = isinstance(expr.right, JoinQuery)

        # If neither side is a join, return early with no change.
        if not left_is_join and not right_is_join:
            return None, None

        # Get the frontier operands for this node.
        if left_is_join and right_is_join:
            frontier_operands = [expr.left.left, expr.left.right, expr.right.left, expr.right.right]
        elif left_is_join:
            frontier_operands = [expr.left.left, expr.left.right, expr.right]
        else:
            frontier_operands = [expr.left, expr.right.left, expr.right.right]

        # Enumerate all binary join trees for all operand permutations.
        # We keep generation order and use it as deterministic tie-break.
        candidates: list[QueryExpr] = []
        for perm in permutations(frontier_operands):
            candidates.extend(self._enumerate_binary_join_trees(list(perm)))

        # Calculate cost for each candidate and pick the best.
        scored: list[tuple[float, int, QueryExpr]] = []
        for index, candidate_expr in enumerate(candidates):
            candidate_cost = self._estimate_plan_cost(candidate_expr)
            scored.append((candidate_cost, index, candidate_expr))

        # Sort by cost, then by generation order as tie-breaker.
        scored.sort(key=lambda item: (item[0], item[1]))

        # Get best cost and return the corresponding expression if it's different from the input.
        _, _, best_expr = scored[0]
        if best_expr != expr:
            return best_expr, f"Cost-based join reordering"

        return None, None

    # Enumerate all binary join parenthesizations over an ordered operand list of 3 or 4.
    def _enumerate_binary_join_trees(self, ordered_operands: list[QueryExpr]) -> list[QueryExpr]:
        arity = len(ordered_operands)

        if arity == 3:
            a, b, c = ordered_operands
            return [
                JoinQuery(left=JoinQuery(left=a, right=b), right=c),
                JoinQuery(left=a, right=JoinQuery(left=b, right=c)),
            ]

        if arity == 4:
            a, b, c, d = ordered_operands
            return [
                JoinQuery(left=JoinQuery(left=JoinQuery(left=a, right=b), right=c), right=d),
                JoinQuery(left=JoinQuery(left=a, right=JoinQuery(left=b, right=c)), right=d),
                JoinQuery(left=JoinQuery(left=a, right=b), right=JoinQuery(left=c, right=d)),
                JoinQuery(left=a, right=JoinQuery(left=JoinQuery(left=b, right=c), right=d)),
                JoinQuery(left=a, right=JoinQuery(left=b, right=JoinQuery(left=c, right=d))),
            ]

        return []

    # Estimates the total cost of executing a query plan.
    def _estimate_plan_cost(self, expr: QueryExpr) -> float:
        #TODO 
