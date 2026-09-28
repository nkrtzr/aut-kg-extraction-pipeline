from typing import Any

from backend.extraction.feedback import FeedbackAssessment
from backend.extraction.inspector import InspectionAssessment
from backend.extraction.pipeline import KnowledgeGraphExtractionPipeline
from backend.extraction.refiner import ActionRefinement
from backend.schemas.process_knowledge.text import Instruction, ObservedAction, ObservedActor, ObservedObject, Segment


class Extractor:
    def extract(self, **kwargs: Any) -> Any:
        model = kwargs["response_model"]
        if model is FeedbackAssessment:
            return FeedbackAssessment()
        if model is InspectionAssessment:
            return InspectionAssessment()
        assert model is ActionRefinement
        return ActionRefinement()


def test_pipeline_extracts_graph_from_deterministic_bindings() -> None:
    actor = ObservedActor(id="actor", name="operator")
    object_ = ObservedObject(id="tool", name="lighter", object_type="tool")
    action = ObservedAction(id="action", start_time_ms=0, end_time_ms=1000, actor_id=actor.id, action="Heat", instrument_id=object_.id)
    instruction = Instruction(id="i", name="Wire repair", segments=[Segment(id="s", scene="", actions=[action], actors=[actor], objects=[object_], uncertainties=[])])
    graph = KnowledgeGraphExtractionPipeline(extractor=Extractor()).extract(instruction)
    graph.validate()
    assert tuple(graph.steps_by_action_id) == ("action",)
