"""Create step-level requirements from resolved instruction objects."""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from backend.extraction.extractor import Extractor
from backend.extraction.resolver import ObjectEntity, ObjectResolver
from backend.extraction.sequnecer import Sequencer
from backend.schemas.process_knowledge.entities import (
    Material,
    PPE,
    ProcessParameter,
    Step,
    Tool,
)
from backend.schemas.process_knowledge.relations import (
    MaterialRequirement,
    PPERequirement,
    ToolRequirement,
)
from backend.schemas.process_knowledge.text import Segment

LevelThreeRelation = ToolRequirement | MaterialRequirement | PPERequirement


class ActionLevelThreeEntities(BaseModel):
    """Level-three entities found in one action."""

    action_id: str
    process_parameters: list[ProcessParameter] = Field(default_factory=list)


class SegmentLevelThreeEntities(BaseModel):
    """Structured LLM result for one analyzed segment."""

    actions: list[ActionLevelThreeEntities] = Field(default_factory=list)


class Refiner:
    """Link resolved physical entities to the steps that reference them."""

    def __init__(
        self,
        sequencer: Sequencer,
        object_resolver: ObjectResolver,
        *,
        extractor: Extractor | None = None,
    ):
        self.sequencer = sequencer
        self.object_resolver = object_resolver
        self.tool_requirements: list[ToolRequirement] = []
        self.material_requirements: list[MaterialRequirement] = []
        self.ppe_requirements: list[PPERequirement] = []
        self.relations: list[LevelThreeRelation] = []
        self.process_parameters: list[ProcessParameter] = []
        self.process_parameters_by_action_id: dict[
            str,
            list[ProcessParameter],
        ] = {}
        self._relations_by_action_id: dict[
            str,
            list[LevelThreeRelation],
        ] = {}
        self._create_relations(object_resolver.entity_bindings)
        self._extract_level_three_entities(extractor or Extractor())

    def _create_relations(
        self,
        bindings: Sequence[tuple[object, ObjectEntity]],
    ) -> None:
        if not bindings and self.object_resolver.objects:
            raise ValueError(
                "ObjectResolver.create_entities() must be called before "
                "initializing Refiner"
            )

        for observed_object, entity in bindings:
            action_ids = getattr(observed_object, "action_ids")
            for action_id in action_ids:
                step = self.sequencer.steps_by_action_id.get(action_id)
                if step is None:
                    raise ValueError(f"Object references unknown action {action_id!r}")

                relation = self._requirement_for(step, entity)
                self.relations.append(relation)
                self._relations_by_action_id.setdefault(
                    action_id,
                    [],
                ).append(relation)
                if isinstance(relation, ToolRequirement):
                    self.tool_requirements.append(relation)
                elif isinstance(relation, MaterialRequirement):
                    self.material_requirements.append(relation)
                else:
                    self.ppe_requirements.append(relation)

    @staticmethod
    def _requirement_for(
        step: Step,
        entity: ObjectEntity,
    ) -> LevelThreeRelation:
        if isinstance(entity, Tool):
            return ToolRequirement(step=step, tool=entity)
        if isinstance(entity, Material):
            return MaterialRequirement(step=step, material=entity)
        if isinstance(entity, PPE):
            return PPERequirement(step=step, ppe=entity)
        raise TypeError(f"Unsupported resolved object entity: {type(entity)!r}")

    def _extract_level_three_entities(self, extractor: Extractor) -> None:
        for segment in self.sequencer.instruction.segments:
            refinement = extractor.extract(
                text=segment.model_dump_json(),
                response_model=SegmentLevelThreeEntities,
                system_prompt=self._level_three_prompt(segment),
            )
            self._add_segment_refinement(segment, refinement)

    def _level_three_prompt(self, segment: Segment) -> str:
        action_ids = [action.id for action in segment.actions]
        return (
            "Analyze this industrial instruction segment and extract explicit "
            "ProcessParameter entities. Associate every parameter with the "
            "exact action_id where it is stated. Return only "
            "action IDs from this list: "
            f"{action_ids!r}. Omit actions with no level-three entities. "
            "Do not infer values or units that are not supported by the "
            "segment. The surrounding instruction is "
            f"id={self.sequencer.instruction.id!r}, "
            f"name={self.sequencer.instruction.name!r}."
        )

    def _add_segment_refinement(
        self,
        segment: Segment,
        refinement: SegmentLevelThreeEntities,
    ) -> None:
        valid_action_ids = {action.id for action in segment.actions}
        returned_action_ids = [
            action_entities.action_id for action_entities in refinement.actions
        ]
        if len(returned_action_ids) != len(set(returned_action_ids)):
            raise ValueError("Segment refinement returned a duplicate action ID")
        if not set(returned_action_ids).issubset(valid_action_ids):
            raise ValueError(
                "Segment refinement returned an action ID outside its segment"
            )

        for action_entities in refinement.actions:
            action_id = action_entities.action_id
            parameters = action_entities.process_parameters
            self.process_parameters_by_action_id[action_id] = parameters
            self.process_parameters.extend(parameters)

            for relation in self._relations_by_action_id.get(action_id, []):
                if isinstance(relation, ToolRequirement) and parameters:
                    relation.parameters = parameters
