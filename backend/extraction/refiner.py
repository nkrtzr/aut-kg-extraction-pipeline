"""Link source-bound physical entities and action-scoped parameters to steps."""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from backend.extraction.extractor import Extractor
from backend.extraction.resolver import ObjectEntity, ObjectResolver
from backend.extraction.sequnecer import Sequencer
from backend.schemas.process_knowledge.entities import Material, ProcessParameter, Step, Tool
from backend.schemas.process_knowledge.relations import MaterialRequirement, PPERequirement, ToolRequirement
from backend.schemas.process_knowledge.text import ActionId, ObservedAction

LevelThreeRelation = ToolRequirement | MaterialRequirement | PPERequirement


class ActionRefinement(BaseModel):
    """Process parameters for the one action supplied to the model."""

    process_parameters: list[ProcessParameter] = Field(default_factory=list)


class Refiner:
    """Link resolved physical entities to source-referenced procedural steps."""

    def __init__(self, sequencer: Sequencer, object_resolver: ObjectResolver, *, extractor: Extractor | None = None):
        self.sequencer = sequencer
        self.object_resolver = object_resolver
        self.tool_requirements: list[ToolRequirement] = []
        self.material_requirements: list[MaterialRequirement] = []
        self.ppe_requirements: list[PPERequirement] = []
        self.relations: list[LevelThreeRelation] = []
        self.process_parameters: list[ProcessParameter] = []
        self.process_parameters_by_action_id: dict[ActionId, list[ProcessParameter]] = {}
        self._relations_by_action_id: dict[ActionId, list[LevelThreeRelation]] = {}
        self._create_relations(object_resolver.entity_bindings)
        self._extract_level_three_entities(extractor or Extractor())

    def _create_relations(self, bindings: Sequence[tuple[object, ObjectEntity]]) -> None:
        if not bindings and self.object_resolver.objects:
            raise ValueError("ObjectResolver.create_entities() must be called before initializing Refiner")
        for observed_object, entity in bindings:
            for action_id in getattr(observed_object, "action_ids"):
                step = self.sequencer.steps_by_action_id.get(action_id)
                if step is None:
                    continue
                relation = self._requirement_for(step, entity)
                self.relations.append(relation)
                self._relations_by_action_id.setdefault(action_id, []).append(relation)
                if isinstance(relation, ToolRequirement):
                    self.tool_requirements.append(relation)
                elif isinstance(relation, MaterialRequirement):
                    self.material_requirements.append(relation)
                else:
                    self.ppe_requirements.append(relation)

    @staticmethod
    def _requirement_for(step: Step, entity: ObjectEntity) -> LevelThreeRelation:
        if isinstance(entity, Tool):
            return ToolRequirement(step=step, tool=entity)
        if isinstance(entity, Material):
            return MaterialRequirement(step=step, material=entity)
        return PPERequirement(step=step, ppe=entity)

    def _extract_level_three_entities(self, extractor: Extractor) -> None:
        for action in self.sequencer.actions:
            refinement = extractor.extract(
                text=action.model_dump_json(),
                response_model=ActionRefinement,
                system_prompt=self._level_three_prompt(action),
            )
            self._add_refinement(action.id, refinement)

    @staticmethod
    def _level_three_prompt(action: ObservedAction) -> str:
        return (
            "Extract only explicit process parameters for this one observed action. "
            "Do not infer values or units. Do not return action IDs or references. "
            f"Current action text={action.action!r}."
        )

    def _add_refinement(self, action_id: ActionId, refinement: ActionRefinement) -> None:
        parameters = refinement.process_parameters
        if not parameters:
            return
        self.process_parameters_by_action_id[action_id] = parameters
        self.process_parameters.extend(parameters)
        for relation in self._relations_by_action_id.get(action_id, []):
            if isinstance(relation, ToolRequirement):
                relation.parameters = parameters
