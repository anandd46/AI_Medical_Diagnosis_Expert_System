
"""
predicate_logic.py
===================
A small but genuine First-Order Predicate Logic engine, built from
scratch, that underlies the medical reasoning in this project.

Concepts implemented
---------------------
- Terms: constants (e.g. "John") and variables (e.g. "?X").
- Predicates: named relations over terms, e.g. HasSymptom(John, Fever).
- Facts: ground predicates (no variables) known to be true.
- Rules: Horn-clause style implications:  P1 AND P2 AND ... -> Q
- Unification: the classic Robinson unification algorithm that finds a
  substitution making two predicates syntactically identical.
- Resolution / forward inference: repeatedly unify rule bodies against
  the known fact base and add newly derivable facts, à la naive forward
  chaining over first-order Horn clauses.
- A simple natural-deduction style proof trace (`explain`) so every
  derived fact can be traced back to the facts/rules that produced it.

This module is domain-agnostic; knowledge_base.py wires it up with
medical vocabulary (HasSymptom, Disease, Treats, ...).

Example
-------
>>> kb = KnowledgeBase()
>>> kb.add_fact(Predicate("HasSymptom", ["John", "Fever"]))
>>> kb.add_fact(Predicate("HasSymptom", ["John", "Cough"]))
>>> kb.add_rule(Rule(
...     body=[Predicate("HasSymptom", ["?X", "Fever"]),
...           Predicate("HasSymptom", ["?X", "Cough"])],
...     head=Predicate("Disease", ["?X", "Flu"])
... ))
>>> kb.forward_chain()
>>> kb.ask(Predicate("Disease", ["John", "Flu"]))
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import get_logger

logger = get_logger(__name__)

Substitution = Dict[str, str]


def is_variable(term: str) -> bool:
    """
    Purpose: Distinguish logic variables from constants by convention:
             any term starting with '?' is a variable (e.g. "?X").
    Input  : term (str)
    Output : bool
    Time Complexity : O(1)
    Space Complexity: O(1)
    """
    return isinstance(term, str) and term.startswith("?")


@dataclass(frozen=True)
class Predicate:
    """
    A predicate is a named relation applied to a tuple of terms.

    Example: Predicate("HasSymptom", ["John", "Fever"])
             represents  HasSymptom(John, Fever)
    """
    name: str
    args: Tuple[str, ...]

    def __init__(self, name: str, args: List[str]):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "args", tuple(args))

    def substitute(self, subst: Substitution) -> "Predicate":
        """
        Purpose: Apply a variable substitution to produce a new predicate.
        Input  : subst - mapping variable-name -> constant/variable value
        Output : a new Predicate with variables replaced where a binding exists
        Time Complexity : O(k) where k = number of arguments
        Space Complexity: O(k)
        """
        new_args = [subst.get(a, a) if is_variable(a) else a for a in self.args]
        return Predicate(self.name, new_args)

    def is_ground(self) -> bool:
        """Return True if the predicate has no variables (i.e. it's a fact)."""
        return not any(is_variable(a) for a in self.args)

    def __str__(self) -> str:
        return f"{self.name}({', '.join(self.args)})"


@dataclass(frozen=True)
class Rule:
    """
    A Horn-clause rule:  body[0] AND body[1] AND ... -> head

    Example: [HasSymptom(?X, Fever), HasSymptom(?X, Cough)] -> Disease(?X, Flu)
    """
    body: Tuple[Predicate, ...]
    head: Predicate
    name: str = ""

    def __init__(self, body: List[Predicate], head: Predicate, name: str = ""):
        object.__setattr__(self, "body", tuple(body))
        object.__setattr__(self, "head", head)
        object.__setattr__(self, "name", name or f"Rule[{head.name}]")

    def __str__(self) -> str:
        body_str = " ^ ".join(str(p) for p in self.body)
        return f"{body_str}  ->  {self.head}"


def unify(p1: Predicate, p2: Predicate, subst: Optional[Substitution] = None) -> Optional[Substitution]:
    """
    Robinson unification algorithm for two predicates.

    Purpose: Find the most general substitution that makes p1 and p2
             syntactically identical, or determine that none exists.
    Input  : p1, p2 - predicates to unify; subst - substitution accumulated so far
    Output : a Substitution dict if unification succeeds, else None
    Logic  : - Names and arities must match.
             - Walk argument pairs; if both constants they must be equal;
               if either is a variable, bind it (respecting any existing
               binding already in subst, resolved recursively).
    Time Complexity : O(k) where k = number of arguments (occurs-check omitted
                       since medical predicates never nest terms).
    Space Complexity: O(k) for the substitution dict.
    """
    if subst is None:
        subst = {}
    if p1.name != p2.name or len(p1.args) != len(p2.args):
        return None

    result = dict(subst)
    for a1, a2 in zip(p1.args, p2.args):
        a1r = _resolve(a1, result)
        a2r = _resolve(a2, result)
        if a1r == a2r:
            continue
        if is_variable(a1r):
            result[a1r] = a2r
        elif is_variable(a2r):
            result[a2r] = a1r
        else:
            return None  # two different constants - cannot unify
    return result


def _resolve(term: str, subst: Substitution) -> str:
    """Follow a chain of substitutions until a constant or unbound variable is reached."""
    seen = set()
    while is_variable(term) and term in subst and term not in seen:
        seen.add(term)
        term = subst[term]
    return term


@dataclass
class InferenceStep:
    """One step in a proof/explanation trace, used by reasoning.py's explainable AI report."""
    rule_name: str
    matched_facts: List[str]
    derived_fact: str


class KnowledgeBase:
    """
    Holds a set of ground facts and Horn-clause rules, and performs
    forward-chaining inference (naive but complete for the small,
    function-free Horn-clause fragment used in this project).
    """

    def __init__(self) -> None:
        self.facts: set[Predicate] = set()
        self.rules: List[Rule] = []
        self.trace: List[InferenceStep] = []

    # ---------------------------------------------------------------- add
    def add_fact(self, predicate: Predicate) -> None:
        """Add a ground fact to the knowledge base. O(1) amortized (set insert)."""
        if not predicate.is_ground():
            raise ValueError(f"Facts must be ground (no variables): {predicate}")
        self.facts.add(predicate)

    def add_rule(self, rule: Rule) -> None:
        """Register an inference rule. O(1)."""
        self.rules.append(rule)

    # ------------------------------------------------------------- query
    def ask(self, predicate: Predicate) -> bool:
        """
        Purpose: Check whether a (possibly non-ground) predicate is
                 entailed by the current fact base.
        Time Complexity : O(F) where F = number of facts (linear scan + unify).
        Space Complexity: O(1) beyond the substitution.
        """
        if predicate.is_ground():
            return predicate in self.facts
        return any(unify(predicate, f) is not None for f in self.facts)

    def query_all(self, predicate: Predicate) -> List[Substitution]:
        """
        Return every substitution that makes `predicate` match a known fact.
        Useful for questions like "what diseases does John have?" via
        Predicate("Disease", ["John", "?D"]).
        Time Complexity : O(F * k)
        """
        results = []
        for fact in self.facts:
            subst = unify(predicate, fact)
            if subst is not None:
                results.append(subst)
        return results

    # ------------------------------------------------------ forward chain
    def forward_chain(self, max_iterations: int = 25) -> List[Predicate]:
        """
        Naive forward chaining: repeatedly try to fire every rule against
        the current fact base, adding newly derived ground facts, until
        no new facts are produced (fixed point) or max_iterations is hit.

        Purpose: Derive every fact entailed by facts + rules.
        Output : list of newly derived facts (in the order they were derived).
        Logic  : For each rule, attempt to unify its body predicates against
                 known facts one at a time, propagating the substitution.
                 If the whole body unifies, instantiate and add the head.
        Time Complexity : O(I * R * F^B) worst case, where I=iterations,
                           R=#rules, F=#facts, B=max body length. In practice
                           B and R are small constants for this project so
                           this is effectively O(I * F).
        Space Complexity: O(F) for the growing fact set.
        """
        newly_derived: List[Predicate] = []
        for _ in range(max_iterations):
            added_this_round = False
            facts_snapshot = list(self.facts)  # freeze to avoid mutation-during-iteration
            for rule in self.rules:
                for subst, matched in self._match_body(rule.body, {}, [], facts_snapshot):
                    derived = rule.head.substitute(subst)
                    if derived.is_ground() and derived not in self.facts:
                        self.facts.add(derived)
                        newly_derived.append(derived)
                        self.trace.append(
                            InferenceStep(
                                rule_name=rule.name,
                                matched_facts=[str(m) for m in matched],
                                derived_fact=str(derived),
                            )
                        )
                        added_this_round = True
            if not added_this_round:
                break
        return newly_derived

    def _match_body(
        self,
        body: Tuple[Predicate, ...],
        subst: Substitution,
        matched: List[Predicate],
        facts_snapshot: List[Predicate],
    ):
        """
        Recursively unify each predicate in a rule body against a fixed
        snapshot of the fact base, threading the substitution through,
        yielding every (final_substitution, matched_facts) combination
        that satisfies the whole conjunction.

        Time Complexity : O(F^B) worst case (B = remaining body length);
                           acceptable since medical rule bodies are short (2-4).
        Space Complexity: O(B) recursion depth.
        """
        if not body:
            yield subst, matched
            return
        first, rest = body[0], body[1:]
        goal = first.substitute(subst)
        for fact in facts_snapshot:
            new_subst = unify(goal, fact, subst)
            if new_subst is not None:
                yield from self._match_body(rest, new_subst, matched + [fact], facts_snapshot)

    # -------------------------------------------------------------- misc
    def explain(self, derived_fact: Predicate) -> Optional[InferenceStep]:
        """Return the inference step (if any) that first produced `derived_fact`."""
        target = str(derived_fact)
        for step in self.trace:
            if step.derived_fact == target:
                return step
        return None

    def reset(self) -> None:
        """Clear all facts, rules and the proof trace (used between diagnosis sessions)."""
        self.facts.clear()
        self.rules.clear()
        self.trace.clear()

    def __len__(self) -> int:
        return len(self.facts)
