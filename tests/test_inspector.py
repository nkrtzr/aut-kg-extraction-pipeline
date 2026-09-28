from typing import Any

from backend.extraction.inspector import InspectionAssessment, Inspector
from backend.extraction.sequnecer import Sequencer
from backend.schemas.process_knowledge.entities import QualityRequirement, Worker
from backend.schemas.process_knowledge.text import Instruction, ObservedAction, ObservedActor, Segment


class Extractor:
    def extract(self, **kwargs: Any) -> InspectionAssessment:
        assert kwargs["response_model"] is InspectionAssessment
        return InspectionAssessment(is_quality_check=True, requirement=QualityRequirement(name="No cracks", requirement_type="visual"))


def test_inspection_binds_assessment_to_current_action() -> None:
    actor = ObservedActor(id="actor", name="operator")
    action = ObservedAction(id="action", start_time_ms=0, end_time_ms=1, actor_id=actor.id, action="Inspect")
    instruction = Instruction(id="i", name="i", segments=[Segment(id="s", scene="", actions=[action], actors=[actor], objects=[], uncertainties=[])])
    inspector = Inspector(Sequencer(instruction), [(actor, Worker(name="operator"))], extractor=Extractor())
    assert inspector.inspection_action_ids == ("action",)
    assert inspector.quality_checks[0].step_execution is inspector.executions_by_action_id["action"]
