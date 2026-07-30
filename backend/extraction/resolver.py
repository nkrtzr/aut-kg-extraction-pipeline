import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel

from backend.extraction.extractor import Extractor
from backend.schemas.process_knowledge.entities import Material, PPE, Tool, Worker

if TYPE_CHECKING:
    from backend.schemas.process_knowledge.text import (
        Instruction,
        ObservedActor,
        ObservedObject,
    )

EntityT = TypeVar("EntityT")
SourceT = TypeVar("SourceT")


class EquivalenceGroup(BaseModel):
    """Input values that refer to the same semantic entity."""

    canonical_entity: str
    entities: list[str]


class SemanticResolution(BaseModel):
    """LLM response containing a partition of all supplied entity values."""

    groups: list[EquivalenceGroup]


ObjectEntity: TypeAlias = Tool | Material | PPE


class ClassifiedObject(BaseModel):
    """A resolved object classified as one supported knowledge entity."""

    input_label: str
    entity_type: Literal["tool", "material", "ppe"]
    tool: Tool | None = None
    material: Material | None = None
    ppe: PPE | None = None


class Resolver(ABC, Generic[EntityT, SourceT]):
    """Base resolver for logical and LLM-assisted semantic deduplication."""

    def __init__(self, source: SourceT | None = None):
        self._entities: list[EntityT] = []
        self._source: SourceT | None = None
        if source is not None:
            self.add_source(source)

    def add(self, entity: EntityT) -> EntityT:
        for existing in self._entities:
            if self._matches(existing, entity):
                self._invalidate_derived_state()
                merge = getattr(existing, "merge", None)
                if callable(merge):
                    merge(entity)
                return existing

        self._invalidate_derived_state()
        self._entities.append(entity)
        return entity

    def _invalidate_derived_state(self) -> None:
        """Invalidate cached outputs derived from the resolver's entities."""

    def add_source(self, source: SourceT) -> tuple[EntityT, ...]:
        """Add all entities exposed by a resolver-specific source."""

        self._invalidate_derived_state()
        self._source = source
        for entity in self.entities_from_source(source):
            self.add(entity)
        return self.entities

    def resolve(self, source: SourceT) -> tuple[EntityT, ...]:
        """Logically resolve entities from a source into this resolver."""

        return self.add_source(source)

    @abstractmethod
    def entities_from_source(self, source: SourceT) -> Sequence[EntityT]:
        """Return entities from the source in observation order."""

    @abstractmethod
    def semantic_label(self, entity: EntityT) -> str:
        """Return the entity representation supplied to the LLM."""

    @property
    @abstractmethod
    def entity_kind(self) -> str:
        """Human-readable entity type used in the semantic prompt."""

    def semantic_deduplicate(
        self,
        entities: list[EntityT] | None = None,
        *,
        extractor: Extractor | None = None,
    ) -> list[EntityT]:
        """Use an LLM to merge values that refer to the same semantic entity.

        The logical entity list is used by default and is not mutated.
        """

        candidates = entities if entities is not None else self._entities
        labels = [self.semantic_label(entity) for entity in candidates]
        if len(labels) != len(set(labels)):
            raise ValueError(
                "Semantic labels must be unique after logical deduplication"
            )
        if not candidates:
            return []

        semantic_extractor = extractor or Extractor()
        resolution = semantic_extractor.extract(
            text=json.dumps(labels, ensure_ascii=False),
            response_model=SemanticResolution,
            system_prompt=self._semantic_deduplication_prompt(),
        )
        canonical_labels = self._validate_semantic_resolution(
            labels,
            resolution,
        )
        self._invalidate_derived_state()
        entities_by_label = dict(zip(labels, candidates, strict=True))
        for group in resolution.groups:
            canonical = entities_by_label[group.canonical_entity]
            merge = getattr(canonical, "merge", None)
            if callable(merge):
                for label in group.entities:
                    if label != group.canonical_entity:
                        merge(entities_by_label[label])
        return [entities_by_label[label] for label in canonical_labels]

    def _semantic_deduplication_prompt(self) -> str:
        return (
            f"You resolve {self.entity_kind} identity in industrial process "
            "instructions. The user provides a JSON list of logically unique "
            f"{self.entity_kind} labels. Group labels only when they "
            f"semantically refer to the same {self.entity_kind}. Every input "
            "label must occur in exactly one group. For each group, choose "
            "canonical_entity verbatim from that group's input labels. Do not "
            "add, remove, translate, or rewrite labels. A label with no "
            "equivalent must be returned as a one-item group. "
            f"{self.source_context()}"
        )

    def source_context(self) -> str:
        """Format source metadata for inclusion in every LLM prompt."""

        if self._source is None:
            return "No source instruction metadata is available."

        metadata = {
            "id": getattr(self._source, "id", None),
            "name": getattr(self._source, "name", None),
        }
        return (
            f"Source instruction metadata: {json.dumps(metadata, ensure_ascii=False)}."
        )

    @staticmethod
    def _validate_semantic_resolution(
        labels: list[str],
        resolution: SemanticResolution,
    ) -> list[str]:
        input_labels = set(labels)
        resolved_labels = [
            entity for group in resolution.groups for entity in group.entities
        ]

        if len(resolved_labels) != len(labels) or set(resolved_labels) != input_labels:
            raise ValueError(
                "The semantic resolution must contain every input entity exactly once"
            )

        canonical_by_entity: dict[str, str] = {}
        for group in resolution.groups:
            if not group.entities or group.canonical_entity not in group.entities:
                raise ValueError(
                    "Each semantic group must choose one of its entities as "
                    "the canonical entity"
                )
            for entity in group.entities:
                canonical_by_entity[entity] = group.canonical_entity

        # Preserve the first-observed order of the semantic groups.
        return list(dict.fromkeys(canonical_by_entity[label] for label in labels))

    @staticmethod
    def _matches(existing: Any, entity: Any) -> bool:
        matches = getattr(existing, "matches", None)
        if callable(matches):
            return bool(matches(entity))
        return bool(existing == entity)

    @property
    def entities(self) -> tuple[EntityT, ...]:
        """Return a read-only view of the resolved collection."""

        return tuple(self._entities)


class ActorResolver(Resolver["ObservedActor", "Instruction"]):
    """Resolve actor references found in an instruction."""

    @property
    def entity_kind(self) -> str:
        return "actor"

    def entities_from_source(
        self,
        instruction: "Instruction",
    ) -> tuple["ObservedActor", ...]:
        return tuple(
            actor
            for segment_actors in instruction.actors.values()
            for actor in segment_actors
        )

    def semantic_label(self, actor: "ObservedActor") -> str:
        return actor.name

    def create_workers(
        self,
        actors: list["ObservedActor"] | None = None,
        *,
        extractor: Extractor | None = None,
    ) -> list[Worker]:
        """Create and, where names allow it, enrich workers for actor names."""

        candidates = actors if actors is not None else self.actors
        actor_names = [actor.name for actor in candidates]
        if len(actor_names) != len(set(actor_names)):
            raise ValueError("Actors must be de-duplicated before creating workers")
        if not actor_names:
            return []

        worker_extractor = extractor or Extractor()
        workers = worker_extractor.extract_list(
            text=json.dumps(actor_names, ensure_ascii=False),
            item_model=Worker,
            system_prompt=self._worker_creation_prompt(),
        )
        return self._validate_workers(actor_names, workers)

    def _worker_creation_prompt(self) -> str:
        return (
            "Create exactly one Worker for every actor name in the supplied "
            "JSON list. Copy each input actor name verbatim into Worker.name; "
            "do not add, remove, translate, merge, or rename workers. Infer "
            "optional attributes such as role only when the actor name itself "
            "provides sufficient evidence. Otherwise leave optional fields "
            "null. Do not invent expertise levels or certifications. "
            f"{self.source_context()}"
        )

    @staticmethod
    def _validate_workers(
        actor_names: list[str],
        workers: list[Worker],
    ) -> list[Worker]:
        worker_names = [worker.name for worker in workers]
        if len(worker_names) != len(actor_names) or set(worker_names) != set(
            actor_names
        ):
            raise ValueError(
                "Worker extraction must return every actor exactly once "
                "without changing its name"
            )

        workers_by_name = {worker.name: worker for worker in workers}
        return [workers_by_name[name] for name in actor_names]

    def add_instruction(
        self,
        instruction: "Instruction",
    ) -> tuple["ObservedActor", ...]:
        """Backward-compatible alias for adding an instruction."""

        return self.add_source(instruction)

    @property
    def actors(self) -> tuple["ObservedActor", ...]:
        """Actor-specific alias for the generic entity collection."""

        return self.entities


class ObjectResolver(Resolver["ObservedObject", "Instruction"]):
    """Resolve objects observed across an instruction's segments."""

    def __init__(self, instruction: "Instruction | None" = None):
        self._entity_bindings: tuple[tuple["ObservedObject", ObjectEntity], ...] = ()
        super().__init__(instruction)

    def _invalidate_derived_state(self) -> None:
        self._entity_bindings = ()

    @property
    def entity_kind(self) -> str:
        return "object"

    def entities_from_source(
        self,
        instruction: "Instruction",
    ) -> tuple["ObservedObject", ...]:
        return tuple(
            observed_object
            for segment_objects in instruction.objects.values()
            for observed_object in segment_objects
        )

    def semantic_label(self, observed_object: "ObservedObject") -> str:
        return f"name={observed_object.name!r}, type={observed_object.object_type!r}"

    def create_entities(
        self,
        objects: Sequence["ObservedObject"] | None = None,
        *,
        extractor: Extractor | None = None,
    ) -> list[ObjectEntity]:
        """Classify resolved objects as Tool, Material, or PPE entities."""

        candidates = objects if objects is not None else self.objects
        labels = [
            self.semantic_label(observed_object) for observed_object in candidates
        ]
        if len(labels) != len(set(labels)):
            raise ValueError("Objects must be de-duplicated before creating entities")
        if not candidates:
            self._entity_bindings = ()
            return []

        entity_extractor = extractor or Extractor()
        classified_objects = entity_extractor.extract_list(
            text=json.dumps(labels, ensure_ascii=False),
            item_model=ClassifiedObject,
            system_prompt=self._object_entity_creation_prompt(),
        )
        entities = self._validate_classified_objects(
            candidates,
            labels,
            classified_objects,
        )
        self._entity_bindings = tuple(zip(candidates, entities, strict=True))
        return entities

    def _object_entity_creation_prompt(self) -> str:
        return (
            "Classify every object in the supplied JSON list as exactly one "
            "of tool, material, or ppe, and create the corresponding Tool, "
            "Material, or PPE value. Return exactly one ClassifiedObject per "
            "input label and copy that label verbatim to input_label. Populate "
            "only the field selected by entity_type and leave the other two "
            "null. Copy the object's original name, without the name= syntax "
            "or quotes, to the nested entity's name. Infer optional attributes "
            "only when the supplied object name or type provides sufficient "
            "evidence; otherwise leave them null. Do not invent manufacturers, "
            "models, specifications, dimensions, or standards. "
            f"{self.source_context()}"
        )

    @staticmethod
    def _validate_classified_objects(
        objects: list["ObservedObject"],
        labels: list[str],
        classified_objects: list[ClassifiedObject],
    ) -> list[ObjectEntity]:
        result_by_label: dict[str, ObjectEntity] = {}

        for classified in classified_objects:
            selected = {
                "tool": classified.tool,
                "material": classified.material,
                "ppe": classified.ppe,
            }
            entity = selected[classified.entity_type]
            populated_fields = [
                value for value in selected.values() if value is not None
            ]
            if entity is None or len(populated_fields) != 1:
                raise ValueError(
                    "Each classified object must populate exactly the entity "
                    "field selected by entity_type"
                )
            if classified.input_label in result_by_label:
                raise ValueError(
                    "Object entity extraction returned a duplicate input label"
                )
            result_by_label[classified.input_label] = entity

        if set(result_by_label) != set(labels):
            raise ValueError(
                "Object entity extraction must return every object exactly "
                "once without changing its input label"
            )

        entities: list[ObjectEntity] = []
        for observed_object, label in zip(objects, labels, strict=True):
            entity = result_by_label[label]
            if entity.name != observed_object.name:
                raise ValueError("Object entity extraction must preserve object names")
            entities.append(entity)
        return entities

    def add_instruction(
        self,
        instruction: "Instruction",
    ) -> tuple["ObservedObject", ...]:
        """Convenience alias for adding an instruction."""

        return self.add_source(instruction)

    @property
    def objects(self) -> tuple["ObservedObject", ...]:
        """Object-specific alias for the generic entity collection."""

        return self.entities

    @property
    def entity_bindings(
        self,
    ) -> tuple[tuple["ObservedObject", ObjectEntity], ...]:
        """Created entities paired with their reference-bearing observations."""

        return self._entity_bindings
