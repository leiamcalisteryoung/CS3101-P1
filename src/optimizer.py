from builder import ProgramState
from cost_optimizer import CostBasedJoinOptimizer
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
    inline_final_query,
    predicate_attributes,
)


class QueryOptimizer:
    # Main entry point
    def run(self, state: ProgramState) -> str:
        # Store state so helper methods can look up relvar info.
        self._state = state

        # Get the final query expression with all LETs inlined
        expr = inline_final_query(state.queries)
        steps: list[str] = [f"Initial inlined query: {format_query_expr(expr)}"]

        # Phase 1: heuristic/equivalence rewrites.
        # Repeatedly apply one rewrite per pass until no rule applies.
        while True:
            rewritten, rule_name = self._rewrite_once_bottom_up(expr)
            if rule_name is None:
                break # no more rewrites, we're done

            # Update expression and record step.
            expr = rewritten
            steps.append(f"{rule_name}: {format_query_expr(expr)}")

        # Phase 2: one bottom-up sweep for cost-based join-chain optimization.
        expr, phase2_steps = CostBasedJoinOptimizer(self._state).optimize(expr)
        steps.extend(phase2_steps)

        steps.append(format_query_expr(expr))

        return "\n".join(steps)

    # Apply first applicable rewrite rule in a bottom-up traversal, returning the rewritten query and the name of the rule applied (or None if no rewrite).
    def _rewrite_once_bottom_up(self, expr: QueryExpr) -> tuple[QueryExpr, str | None]:
        # Match on query type and recursively rewrite children
        # If any child rewrites, return a new query node with the rewritten child and the rule_name
        if isinstance(expr, SelectQuery):
            new_source, rule_name = self._rewrite_once_bottom_up(expr.source)
            if rule_name is not None:
                return SelectQuery(source=new_source, predicate=expr.predicate), rule_name

        elif isinstance(expr, ProjectQuery):
            new_source, rule_name = self._rewrite_once_bottom_up(expr.source)
            if rule_name is not None:
                return ProjectQuery(source=new_source, attributes=list(expr.attributes)), rule_name

        elif isinstance(expr, RenameQuery):
            new_source, rule_name = self._rewrite_once_bottom_up(expr.source)
            if rule_name is not None:
                return RenameQuery(source=new_source, new_attributes=list(expr.new_attributes)), rule_name

        elif isinstance(expr, UnionQuery):
            new_left, rule_name = self._rewrite_once_bottom_up(expr.left)
            if rule_name is not None:
                return UnionQuery(left=new_left, right=expr.right), rule_name

            new_right, rule_name = self._rewrite_once_bottom_up(expr.right)
            if rule_name is not None:
                return UnionQuery(left=expr.left, right=new_right), rule_name

        elif isinstance(expr, DifferenceQuery):
            new_left, rule_name = self._rewrite_once_bottom_up(expr.left)
            if rule_name is not None:
                return DifferenceQuery(left=new_left, right=expr.right), rule_name

            new_right, rule_name = self._rewrite_once_bottom_up(expr.right)
            if rule_name is not None:
                return DifferenceQuery(left=expr.left, right=new_right), rule_name

        elif isinstance(expr, JoinQuery):
            new_left, rule_name = self._rewrite_once_bottom_up(expr.left)
            if rule_name is not None:
                return JoinQuery(left=new_left, right=expr.right), rule_name

            new_right, rule_name = self._rewrite_once_bottom_up(expr.right)
            if rule_name is not None:
                return JoinQuery(left=expr.left, right=new_right), rule_name

        # Base case: no child changed, try all node-level rewrites here.
        rewritten, rule_name = self._apply_node_rewrite_rules(expr)
        if rewritten is not None:
            return rewritten, rule_name

        # No rewrite applied at this node
        return expr, None

    # Try all rewrites for one node in priority order, returning the first successful rewrite and its rule name (or None if no rewrite applies).
    def _apply_node_rewrite_rules(self, expr: QueryExpr) -> tuple[QueryExpr | None, str | None]:
        # 1) Try trivial simplifications first.
        rewritten = self._apply_trivial_simplification(expr)
        if rewritten is not None and rewritten != expr:
            return rewritten, "Trivial simplification"

        # 2) Unary equivalences
        rewritten, rule_name = self._apply_unary_equivalences(expr)
        if rewritten is not None:
            return rewritten, rule_name
        
        # 3) Pushing selections
        rewritten, rule_name = self._apply_selection_pushdown(expr)
        if rewritten is not None:
            return rewritten, rule_name
        
        # 4) Pushing projections
        rewritten, rule_name = self._apply_projection_pushdown(expr)
        if rewritten is not None:
            return rewritten, rule_name

        # No rewrite applied to this node
        return None, None
    
    # Trivial simplifications that don't require knowing the relation contents, just the query structure.
    def _apply_trivial_simplification(self, expr: QueryExpr) -> QueryExpr | None:
        # r ∪ r ≡ r
        if isinstance(expr, UnionQuery) and expr.left == expr.right:
            return expr.left

        # r ▷◁ r ≡ r
        if isinstance(expr, JoinQuery) and expr.left == expr.right:
            return expr.left

        # r − r ≡ ∅
        if isinstance(expr, DifferenceQuery) and expr.left == expr.right:
            return EmptyQuery()

        # r ∪ ∅ ≡ r,  ∅ ∪ r ≡ r
        if isinstance(expr, UnionQuery):
            if isinstance(expr.left, EmptyQuery):
                return expr.right
            if isinstance(expr.right, EmptyQuery):
                return expr.left

        # r ▷◁ ∅ ≡ ∅,  ∅ ▷◁ r ≡ ∅
        if isinstance(expr, JoinQuery):
            if isinstance(expr.left, EmptyQuery) or isinstance(expr.right, EmptyQuery):
                return EmptyQuery()

        # r − ∅ ≡ r,  ∅ − r ≡ ∅
        if isinstance(expr, DifferenceQuery):
            if isinstance(expr.right, EmptyQuery):
                return expr.left
            if isinstance(expr.left, EmptyQuery):
                return EmptyQuery()

        # πA(∅) ≡ ∅
        if isinstance(expr, ProjectQuery) and isinstance(expr.source, EmptyQuery):
            return EmptyQuery()

        # σθ(∅) ≡ ∅
        if isinstance(expr, SelectQuery) and isinstance(expr.source, EmptyQuery):
            return EmptyQuery()

        # ρN(∅) ≡ ∅
        if isinstance(expr, RenameQuery) and isinstance(expr.source, EmptyQuery):
            return EmptyQuery()

        return None
    
    # Apply unary equivalence rewrites that only require structural pattern matching, without needing to know relation contents.
    def _apply_unary_equivalences(self, expr: QueryExpr) -> tuple[QueryExpr | None, str | None]:
        # σθ1 (σθ2 (r)) ≡ σθ1∧θ2 (r)
        if isinstance(expr, SelectQuery) and isinstance(expr.source, SelectQuery):
            merged = SelectQuery(
                source=expr.source.source,
                predicate=AndPredicate(left=expr.predicate, right=expr.source.predicate),
            )
            return merged, "Selection conjunction merge"
        
        # σθ1 (σθ2 (r)) ≡ σθ2 (σθ1 (r)) is pointless to apply because the predicates will be combined anyway

        # πA1 (πA2 (r)) ≡ πA1 (r)
        if isinstance(expr, ProjectQuery) and isinstance(expr.source, ProjectQuery):
            merged = ProjectQuery(
                source=expr.source.source,
                attributes=[attr for attr in expr.attributes],
            )
            return merged, "Nested projection merge"
        
        # ρa1 (ρa2 (r)) ≡ ρa3 (r) (USQL only has generalised renaming so a3 = a1 here)
        if isinstance(expr, RenameQuery) and isinstance(expr.source, RenameQuery):
            merged = RenameQuery(
                source=expr.source.source,
                new_attributes=[attr for attr in expr.new_attributes],
            )
            return merged, "Nested rename merge"

        return None, None

    # Apply rewrites that push selections down through other operators, returning the rewritten query and the rule name (or None if no rewrite applies).
    def _apply_selection_pushdown(self, expr: QueryExpr) -> tuple[QueryExpr | None, str | None]:
        # σθ(πA(r)) ≡ πA(σθ(r))
        if isinstance(expr, SelectQuery) and isinstance(expr.source, ProjectQuery):
            pushed = ProjectQuery(
                source=SelectQuery(source=expr.source.source, predicate=expr.predicate),
                attributes=list(expr.source.attributes),
            )
            return pushed, "Selection pushdown through projection"
        
        # σθ(ρa(r)) ≡ ρa(σθ′ (r)) where θ′ is an appropriate renaming of θ
        if isinstance(expr, SelectQuery) and isinstance(expr.source, RenameQuery):
            # The predicate currently refers to the renamed attribute names.
            # Create mapping of new to old names so we can revert to the original names in the predicate
            old_attrs = self._output_attributes(expr.source.source)
            renamed_attrs = list(expr.source.new_attributes)
            new_to_old = dict(zip(renamed_attrs, old_attrs))
            adjusted_predicate = self._rename_predicate_attributes(expr.predicate, new_to_old)

            pushed = RenameQuery(
                source=SelectQuery(source=expr.source.source, predicate=adjusted_predicate),
                new_attributes=list(expr.source.new_attributes),
            )
            return pushed, "Selection pushdown through rename"
        
        # σθ(r ∪s) ≡ σθ(r) ∪σθ(s)
        if isinstance(expr, SelectQuery) and isinstance(expr.source, UnionQuery):
            pushed = UnionQuery(
                left=SelectQuery(source=expr.source.left, predicate=expr.predicate),
                right=SelectQuery(source=expr.source.right, predicate=expr.predicate),
            )
            return pushed, "Selection pushdown through union"
        
        # σθ(r−s) ≡ σθ(r)−σθ(s)
        if isinstance(expr, SelectQuery) and isinstance(expr.source, DifferenceQuery):
            pushed = DifferenceQuery(
                left=SelectQuery(source=expr.source.left, predicate=expr.predicate),
                right=SelectQuery(source=expr.source.right, predicate=expr.predicate),
            )
            return pushed, "Selection pushdown through difference"
        
        # σθ1∧θ2∧θ3 (r ▷◁s) ≡ σθ1 (σθ2 (r) ▷◁σθ3 (s)) where:
        #   θ1:  all referenced attrs are output by r, pushed into r
        #   θ2:  all referenced attrs are output by s, pushed into s
        #   θ3: predicate spans attrs from both sides, stays outside the join
        if isinstance(expr, SelectQuery) and isinstance(expr.source, JoinQuery):
            left_attrs  = set(self._output_attributes(expr.source.left))
            right_attrs = set(self._output_attributes(expr.source.right))

            # Flatten the possibly nested predicate tree into individual atomic predicates
            atomic_preds = self._flatten_conjunction(expr.predicate)

            left_preds:  list[Predicate] = []  # pushed into left operand
            right_preds: list[Predicate] = []  # pushed into right operand
            cross_preds: list[Predicate] = []  # kept outside (span both sides)

            for pred in atomic_preds:
                attrs = predicate_attributes(pred)
                in_left  = attrs.issubset(left_attrs)
                in_right = attrs.issubset(right_attrs)

                if in_left and in_right:
                    # All referenced attributes are in both relations, push to both
                    left_preds.append(pred)
                    right_preds.append(pred)
                elif in_left:
                    left_preds.append(pred)
                elif in_right:
                    right_preds.append(pred)
                else:
                    # The predicate references one attribute from each side, so it remains outside the join
                    cross_preds.append(pred)

            # Only rewrite if at least one predicate can actually be pushed down.
            if not left_preds and not right_preds:
                return None, None

            # Build the rewritten left and right operands, wrapping with a SelectQuery
            # only when there are predicates to push to that side.
            new_left  = expr.source.left
            new_right = expr.source.right

            if left_preds:
                new_left  = SelectQuery(source=new_left,  predicate=self._build_conjunction(left_preds))
            if right_preds:
                new_right = SelectQuery(source=new_right, predicate=self._build_conjunction(right_preds))

            new_join = JoinQuery(left=new_left, right=new_right)

            # If any cross predicates remain then wrap in a SelectQuery
            if cross_preds:
                result: QueryExpr = SelectQuery(source=new_join, predicate=self._build_conjunction(cross_preds))
            else:
                result = new_join

            return result, "Selection pushdown through join"

        return None, None

    # Apply rewrites that push projections down through other operators, returning the rewritten query and the rule name (or None if no rewrite applies).
    def _apply_projection_pushdown(self, expr: QueryExpr) -> tuple[QueryExpr | None, str | None]:
        # sπA(ρa(r)) ≡ ρa(πA′ (r)) where A' is an appropriate renaming of A
        if isinstance(expr, ProjectQuery) and isinstance(expr.source, RenameQuery):
            # Create mapping of new to old names so we can revert to the original names in the projection list
            old_attrs = self._output_attributes(expr.source.source)
            renamed_attrs = list(expr.source.new_attributes)
            new_to_old = dict(zip(renamed_attrs, old_attrs))
            adjusted_attributes = [new_to_old[attr] for attr in expr.attributes]

            pushed = RenameQuery(
                source=ProjectQuery(source=expr.source.source, attributes=adjusted_attributes),
                new_attributes=list(expr.attributes),
            )
            return pushed, "Projection pushdown through rename"
        
        # πA(σθ(r)) ≡ πA(σθ(πA∪B(r))) is NOT implemented: it would cycle with selection pushdown 
        # through projection and selection should be done first (heuristic)

        # πA(r ∪s) ≡ πA(r) ∪πA(r)
        if isinstance(expr, ProjectQuery) and isinstance(expr.source, UnionQuery):
            pushed = UnionQuery(
                left=ProjectQuery(source=expr.source.left, attributes=list(expr.attributes)),
                right=ProjectQuery(source=expr.source.right, attributes=list(expr.attributes)),
            )
            return pushed, "Projection pushdown through union"

        # πα(r ▷◁ s) ≡ πα(πR′(r) ▷◁ πS′(s)) where R′ = R ∩(α∪S) and S′ = S ∩(α∪R).
        if isinstance(expr, ProjectQuery) and isinstance(expr.source, JoinQuery):
            # Get the output attributes of the left and right sides of the join, and determine which attributes need to be kept from each side to produce the final projection.
            # Use set operations to determine which attributes to keep but lists so it is deterministic.
            left_attr_list = self._output_attributes(expr.source.left)
            right_attr_list = self._output_attributes(expr.source.right)
            left_attrs = set(left_attr_list)
            right_attrs = set(right_attr_list)
            proj_attrs = set(expr.attributes)

            keep_left = proj_attrs.union(right_attrs)
            keep_right = proj_attrs.union(left_attrs)
            new_left_attrs = [attr for attr in left_attr_list if attr in keep_left]
            new_right_attrs = [attr for attr in right_attr_list if attr in keep_right]
            
            # if pushdown would not remove any attribute from either side, do not rewrite.
            if new_left_attrs == left_attr_list and new_right_attrs == right_attr_list:
                return None, None
            
            # Create the inner projections on each side of the join
            if new_left_attrs == left_attr_list:
                new_left = expr.source.left
            else:
                new_left = ProjectQuery(source=expr.source.left, attributes=new_left_attrs)

            if new_right_attrs == right_attr_list:
                new_right = expr.source.right
            else:                
                new_right = ProjectQuery(source=expr.source.right, attributes=new_right_attrs)

            # Create the join and wrap in original projection
            pushed_join = JoinQuery(left=new_left, right=new_right)
            pushed = ProjectQuery(source=pushed_join, attributes=list(expr.attributes))
            return pushed, "Projection pushdown through join"

        return None, None

    # Flatten a predicate tree into a list of atomic predicates by recursively flattening AndPredicates. Leaves will be AttrEqAttrPredicate or AttrEqConstPredicate.
    @staticmethod
    def _flatten_conjunction(pred: Predicate) -> list[Predicate]:
        if isinstance(pred, AndPredicate):
            return (
                QueryOptimizer._flatten_conjunction(pred.left)
                + QueryOptimizer._flatten_conjunction(pred.right)
            )
        # Leaf predicate (AttrEqAttrPredicate or AttrEqConstPredicate)
        return [pred]

    # Combine a list of predicates into a left-associative AndPredicate conjunction.
    @staticmethod
    def _build_conjunction(preds: list[Predicate]) -> Predicate:
        result = preds[0]
        for pred in preds[1:]:
            result = AndPredicate(left=result, right=pred)
        return result

    # Helper method to get the output attributes of a query expression.
    def _output_attributes(self, expr: QueryExpr) -> list[str]:
        # Base case: RelVarQuery, look up the relvar and return its relation's attribute names.
        if isinstance(expr, RelVarQuery):
            relvar = self._state.relvars.get(expr.name)
            if relvar is None:
                raise ValueError(f"Unknown relvar '{expr.name}' while inferring attributes.")
            return relvar.relation.attr_names()
        
        if isinstance(expr, EmptyQuery):
            return []
        
        # Otherwise recursively determine output attributes based on query types.
        if isinstance(expr, SelectQuery):
            return self._output_attributes(expr.source)

        if isinstance(expr, ProjectQuery):
            return list(expr.attributes)

        if isinstance(expr, RenameQuery):
            return list(expr.new_attributes)

        if isinstance(expr, UnionQuery) or isinstance(expr, DifferenceQuery):
            return self._output_attributes(expr.left)

        if isinstance(expr, JoinQuery):
            left_attrs = self._output_attributes(expr.left)
            right_attrs = self._output_attributes(expr.right)
            right_only = [attr for attr in right_attrs if attr not in left_attrs]
            return left_attrs + right_only

        raise ValueError(f"Unsupported query node type: {type(expr).__name__}")

    # Helper method to rename attributes in a predicate according to a mapping, used for selection pushdown through renames.
    def _rename_predicate_attributes(
        self,
        predicate: Predicate,
        new_to_old: dict[str, str],
    ) -> Predicate:
        if isinstance(predicate, AttrEqAttrPredicate):
            return AttrEqAttrPredicate(
                left_attr=new_to_old[predicate.left_attr],
                right_attr=new_to_old[predicate.right_attr],
            )

        if isinstance(predicate, AttrEqConstPredicate):
            return AttrEqConstPredicate(
                attr=new_to_old[predicate.attr],
                value=predicate.value,
            )

        if isinstance(predicate, AndPredicate):
            return AndPredicate(
                left=self._rename_predicate_attributes(predicate.left, new_to_old),
                right=self._rename_predicate_attributes(predicate.right, new_to_old),
            )

        raise ValueError(f"Unsupported predicate type: {type(predicate).__name__}")
