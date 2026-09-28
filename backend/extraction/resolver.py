"""Deterministic actor and object resolution from source observations."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar

from pydantic import BaseModel

from backend.extraction.extractor import Extractor
from backend.schemas.process_knowledge.entities import Material, PPE, Tool, Worker

if TYPE_CHECKING:
    from backend.schemas.process_knowledge.text import Instruction, ObservedActor, ObservedObject

EntityT = TypeVar("EntityT")
SourceT = TypeVar("SourceT")
ObjectEntity: TypeAlias = Tool | Material | PPE


class ObjectClassification(BaseModel):
    """Optional semantic type/subtype assessment for one known object."""

    entity_type: Literal["tool", "material", "ppe"]
    subtype: str | None = None


class Resolver(ABC):
    def __init__(self, source: Any | None = None):
        self._entities: list[Any] = []
        if source is not None:
            self.add_source(source)

    def add(self, entity: Any) -> Any:
        for existing in self._entities:
            matches = getattr(existing, "matches", None)
            if callable(matches) and matches(entity):
                merge = getattr(existing, "merge", None)
                if callable(merge):
                    merge(entity)
                return existing
        self._entities.append(entity)
        return entity

    def add_source(self, source: Any) -> tuple[Any, ...]:
        for entity in self.entities_from_source(source):
            self.add(entity)
        return self.entities

    @abstractmethod
    def entities_from_source(self, source: Any) -> Sequence[Any]: ...

    @property
    def entities(self) -> tuple[Any, ...]:
        return tuple(self._entities)


class ActorResolver(Resolver):
    def __init__(self, instruction: "Instruction | None" = None):
        self._worker_bindings: tuple[tuple[Any, Worker], ...] = ()
        super().__init__(instruction)

    def entities_from_source(self, instruction: "Instruction") -> tuple["ObservedActor", ...]:
        return tuple(actor for actors in instruction.actors.values() for actor in actors)

    def create_workers(self, actors: Sequence["ObservedActor"] | None = None, *, extractor: Extractor | None = None) -> list[Worker]:
        candidates = list(actors if actors is not None else self.entities)
        workers = [Worker(name=actor.name, role=self._role_for(actor.name)) for actor in candidates]
        self._worker_bindings = tuple(zip(candidates, workers, strict=True))
        return workers

    @staticmethod
    def _role_for(name: str) -> str | None:
        normalized = name.casefold().replace(" ", "_")
        return normalized if normalized in {"operator", "supervisor", "quality_inspector", "maintenance", "trainee", "welding_engineer"} else None

    def add_instruction(self, instruction: "Instruction") -> tuple["ObservedActor", ...]:
        return self.add_source(instruction)

    @property
    def actors(self) -> tuple["ObservedActor", ...]:
        return self.entities

    @property
    def worker_bindings(self) -> tuple[tuple["ObservedActor", Worker], ...]:
        return self._worker_bindings


class ObjectResolver(Resolver):
    def __init__(self, instruction: "Instruction | None" = None):
        self._entity_bindings: tuple[tuple[Any, ObjectEntity], ...] = ()
        super().__init__(instruction)

    def entities_from_source(self, instruction: "Instruction") -> tuple["ObservedObject", ...]:
        return tuple(obj for objects in instruction.objects.values() for obj in objects)

    def create_entities(self, objects: Sequence["ObservedObject"] | None = None, *, extractor: Extractor | None = None) -> list[ObjectEntity]:
        candidates = list(objects if objects is not None else self.entities)
        entities = [self._entity_for(object_, extractor) for object_ in candidates]
        self._entity_bindings = tuple(zip(candidates, entities, strict=True))
        return entities

    def _entity_for(self, observed_object: "ObservedObject", extractor: Extractor | None) -> ObjectEntity:
        source_type = observed_object.object_type.casefold()
        if source_type == "tool":
            return Tool(name=observed_object.name, tool_type=observed_object.object_type)
        if source_type in {"material", "assembly"}:
            return Material(name=observed_object.name, material_type=observed_object.object_type)
        if source_type in {"ppe", "protective equipment"}:
            return PPE(name=observed_object.name, ppe_type=observed_object.object_type)
        assessment = (extractor or Extractor()).extract(
            text=observed_object.model_dump_json(),
            response_model=ObjectClassification,
            system_prompt=("Classify this one known object as tool, material, or ppe. " "Return only entity_type and optional subtype; do not return names or IDs."),
        )
        if assessment.entity_type == "tool":
            return Tool(name=observed_object.name, tool_type=assessment.subtype)
        if assessment.entity_type == "ppe":
            return PPE(name=observed_object.name, ppe_type=assessment.subtype)
        return Material(name=observed_object.name, material_type=assessment.subtype)

    def add_instruction(self, instruction: "Instruction") -> tuple["ObservedObject", ...]:
        return self.add_source(instruction)

    @property
    def objects(self) -> tuple["ObservedObject", ...]:
        return self.entities

    @property
    def entity_bindings(self) -> tuple[tuple["ObservedObject", ObjectEntity], ...]:
        return self._entity_bindings
