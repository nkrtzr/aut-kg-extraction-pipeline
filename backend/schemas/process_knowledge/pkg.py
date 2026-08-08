"""In-memory aggregate for an extracted procedural knowledge graph."""

from collections.abc import Iterable, Mapping
from typing import Literal, Self

from pydantic import BaseModel

from backend.schemas.base import KGEntity, KGRelation
from backend.schemas.process_knowledge.entities import (
    Procedure,
    ProcessParameter,
    Step,
)
from backend.schemas.process_knowledge.relations import StepOrder
from backend.schemas.process_knowledge.text import ActionId


DependentPolicy = Literal["cascade", "restrict"]


class ProceduralKnowledgeGraph:
    """Own the entities, relations, and indexes of one procedural KG.

    Pipeline stages add their results through this class instead of maintaining
    another mutable copy of the graph. Action IDs are retained as stable source
    identifiers because step numbers change when a step is removed.
    """

    def __init__(self) -> None:
        self._entities: list[KGEntity] = []
        self._relations: list[KGRelation] = []
        self.procedure: Procedure | None = None
        self.steps_by_action_id: dict[ActionId, Step] = {}
        self.process_parameters_by_action_id: dict[
            ActionId, list[ProcessParameter]
        ] = {}

    @classmethod
    def from_results(
        cls,
        *,
        entities: Iterable[KGEntity],
        procedure: Procedure,
        steps_by_action_id: Mapping[ActionId, Step],
        process_parameters_by_action_id: Mapping[
            ActionId, Iterable[ProcessParameter]
        ]
        | None = None,
        relations: Iterable[KGRelation] = (),
    ) -> Self:
        """Assemble and validate a graph from completed extraction outputs.

        The insertion order is centralized here: independent entities and
        sequence endpoints are registered before parameters, and relations are
        added only after all of their entity endpoints exist.
        """

        graph = cls()
        graph.add_entities(entities)
        graph.add_sequence(procedure, steps_by_action_id)
        parameters_by_action = process_parameters_by_action_id or {}
        for action_id, parameters in parameters_by_action.items():
            graph.add_process_parameters(action_id, parameters)
        graph.add_relations(relations)
        graph.validate()
        return graph

    @property
    def entities(self) -> tuple[KGEntity, ...]:
        """Return a read-only view of the graph's entities."""

        return tuple(self._entities)

    @property
    def relations(self) -> tuple[KGRelation, ...]:
        """Return a read-only view of the graph's relations."""

        return tuple(self._relations)

    @property
    def steps(self) -> tuple[Step, ...]:
        """Return steps in their current procedural order."""

        return tuple(self.steps_by_action_id.values())

    @property
    def step_orders(self) -> tuple[StepOrder, ...]:
        """Return the current adjacent step-order relations."""

        return tuple(
            relation
            for relation in self._relations
            if isinstance(relation, StepOrder)
        )

    def add_entity(self, entity: KGEntity) -> None:
        """Register an entity, rejecting the same object being added twice."""

        if self._contains_identity(self._entities, entity):
            raise ValueError("Entity object is already present in the graph")
        self._entities.append(entity)

    def add_entities(self, entities: Iterable[KGEntity]) -> None:
        """Register multiple entities."""

        for entity in entities:
            self.add_entity(entity)

    def add_relation(self, relation: KGRelation) -> None:
        """Register a relation after checking that its entity endpoints exist."""

        if self._contains_identity(self._relations, relation):
            raise ValueError("Relation object is already present in the graph")

        missing = [
            entity
            for entity in self._referenced_entities(relation)
            if not self._contains_identity(self._entities, entity)
        ]
        if missing:
            raise ValueError(
                "Relation references entities that are not in the graph: "
                f"{[entity.name for entity in missing]!r}"
            )
        self._relations.append(relation)

    def add_relations(self, relations: Iterable[KGRelation]) -> None:
        """Register multiple relations."""

        for relation in relations:
            self.add_relation(relation)

    def add_procedure(self, procedure: Procedure) -> None:
        """Register the graph's single procedure."""

        if self.procedure is not None:
            raise ValueError("The graph already has a procedure")
        self.add_entity(procedure)
        self.procedure = procedure

    def add_step(self, action_id: ActionId, step: Step) -> None:
        """Append a step and bind it to its stable source action ID."""

        if action_id in self.steps_by_action_id:
            raise ValueError(f"A step already exists for action {action_id!r}")
        if self._contains_identity(self.steps_by_action_id.values(), step):
            raise ValueError("Step object is already bound to another action")

        self.add_entity(step)
        self.steps_by_action_id[action_id] = step
        self._rebuild_sequence()

    def add_sequence(
        self,
        procedure: Procedure,
        steps_by_action_id: Mapping[ActionId, Step],
    ) -> None:
        """Add a sequencer result while preserving mapping insertion order."""

        self.add_procedure(procedure)
        for action_id, step in steps_by_action_id.items():
            self.add_step(action_id, step)

    def add_process_parameters(
        self,
        action_id: ActionId,
        parameters: Iterable[ProcessParameter],
    ) -> None:
        """Register process parameters associated with an existing step."""

        if action_id not in self.steps_by_action_id:
            raise ValueError(f"No step exists for action {action_id!r}")
        if action_id in self.process_parameters_by_action_id:
            raise ValueError(f"Parameters already exist for action {action_id!r}")

        parameter_list = list(parameters)
        self.add_entities(parameter_list)
        self.process_parameters_by_action_id[action_id] = parameter_list

    def remove_step(
        self,
        action_id: ActionId,
        *,
        dependent_policy: DependentPolicy = "cascade",
    ) -> Step:
        """Remove a step and consistently update all derived graph structures.

        ``restrict`` refuses removal when non-order relations refer to the step.
        ``cascade`` removes those relations, parameters owned by the action,
        renumbers the remaining steps, and rebuilds adjacent StepOrder edges.
        """

        try:
            step = self.steps_by_action_id[action_id]
        except KeyError as error:
            raise KeyError(f"No step exists for action {action_id!r}") from error

        dependent_relations = [
            relation
            for relation in self._relations
            if not isinstance(relation, StepOrder)
            and self._references_identity(relation, step)
        ]
        dependent_parameters = self.process_parameters_by_action_id.get(action_id, [])
        if dependent_policy == "restrict" and (
            dependent_relations or dependent_parameters
        ):
            raise ValueError(
                f"Step for action {action_id!r} has "
                f"{len(dependent_relations)} dependent relation(s) and "
                f"{len(dependent_parameters)} parameter(s)"
            )

        del self.steps_by_action_id[action_id]
        self._remove_identity(self._entities, step)
        if dependent_policy == "cascade":
            self._relations = [
                relation
                for relation in self._relations
                if not self._references_identity(relation, step)
            ]

        removed_parameters = self.process_parameters_by_action_id.pop(action_id, [])
        for parameter in removed_parameters:
            if not any(
                self._contains_identity(parameters, parameter)
                for parameters in self.process_parameters_by_action_id.values()
            ):
                self._remove_identity(self._entities, parameter)

        self._rebuild_sequence()
        self.validate()
        return step

    def validate(self) -> None:
        """Validate indexes, numbering, ordering, and relation endpoints."""

        steps = list(self.steps_by_action_id.values())
        if any(not self._contains_identity(self._entities, step) for step in steps):
            raise ValueError("The step index contains a step outside the graph")
        if [step.step_number for step in steps] != list(range(1, len(steps) + 1)):
            raise ValueError("Step numbers must be contiguous and one-based")

        expected_pairs = list(zip(steps, steps[1:]))
        actual_orders = self.step_orders
        if len(actual_orders) != len(expected_pairs) or any(
            order.before is not before or order.after is not after
            for order, (before, after) in zip(
                actual_orders, expected_pairs, strict=True
            )
        ):
            raise ValueError("StepOrder relations do not match the step sequence")

        for relation in self._relations:
            if any(
                not self._contains_identity(self._entities, entity)
                for entity in self._referenced_entities(relation)
            ):
                raise ValueError("A relation references an entity outside the graph")

    def _rebuild_sequence(self) -> None:
        steps = list(self.steps_by_action_id.values())
        for number, step in enumerate(steps, start=1):
            step.step_number = number

        self._relations = [
            relation
            for relation in self._relations
            if not isinstance(relation, StepOrder)
        ]
        self._relations.extend(
            StepOrder(before=before, after=after)
            for before, after in zip(steps, steps[1:])
        )

    @classmethod
    def _referenced_entities(cls, value: object) -> list[KGEntity]:
        found: list[KGEntity] = []
        cls._collect_referenced_entities(value, found, is_root=True)
        return found

    @classmethod
    def _collect_referenced_entities(
        cls,
        value: object,
        found: list[KGEntity],
        *,
        is_root: bool = False,
    ) -> None:
        if isinstance(value, KGEntity):
            if not cls._contains_identity(found, value):
                found.append(value)
            return
        if isinstance(value, BaseModel):
            if not is_root and isinstance(value, KGRelation):
                pass
            for field_name in type(value).model_fields:
                cls._collect_referenced_entities(getattr(value, field_name), found)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                cls._collect_referenced_entities(item, found)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                cls._collect_referenced_entities(item, found)

    @classmethod
    def _references_identity(cls, value: object, target: object) -> bool:
        if value is target:
            return True
        if isinstance(value, BaseModel):
            return any(
                cls._references_identity(getattr(value, field_name), target)
                for field_name in type(value).model_fields
            )
        if isinstance(value, Mapping):
            return any(
                cls._references_identity(item, target) for item in value.values()
            )
        if isinstance(value, (list, tuple, set)):
            return any(cls._references_identity(item, target) for item in value)
        return False

    @staticmethod
    def _contains_identity(values: Iterable[object], target: object) -> bool:
        return any(value is target for value in values)

    @staticmethod
    def _remove_identity(values: list[object], target: object) -> None:
        values[:] = [value for value in values if value is not target]
