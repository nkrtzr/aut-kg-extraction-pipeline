from typing import Any

from backend.extraction.refiner import ActionRefinement, Refiner
from backend.extraction.resolver import ObjectResolver
from backend.extraction.sequnecer import Sequencer
from backend.schemas.process_knowledge.entities import ProcessParameter
from backend.schemas.process_knowledge.text import Instruction, ObservedAction, ObservedActor, ObservedObject, Segment


class Extractor:
    def extract(self, **kwargs: Any) -> ActionRefinement:
        assert kwargs["response_model"] is ActionRefinement
        return ActionRefinement(process_parameters=[ProcessParameter(name="Heat", parameter_type="temperature", unit="C", nominal_value=20)])


def test_refinement_binds_parameters_to_current_action() -> None:
    actor = ObservedActor(id="actor", name="operator")
    object_ = ObservedObject(id="tool", name="lighter", object_type="tool")
    action = ObservedAction(id="action", start_time_ms=0, end_time_ms=1, actor_id=actor.id, action="Heat", instrument_id=object_.id)
    instruction = Instruction(id="i", name="i", segments=[Segment(id="s", scene="", actions=[action], actors=[actor], objects=[object_], uncertainties=[])])
    resolver = ObjectResolver(instruction)
    resolver.create_entities()
    refiner = Refiner(Sequencer(instruction), resolver, extractor=Extractor())
    assert tuple(refiner.process_parameters_by_action_id) == ("action",)
    assert refiner.tool_requirements[0].parameters == refiner.process_parameters
