from typing import Any

from backend.extraction.feedback import FeedbackAssessment, FeedbackResolver
from backend.schemas.process_knowledge.text import Instruction, ObservedAction, ObservedActor, Segment


class Extractor:
    def extract(self, **kwargs: Any) -> FeedbackAssessment:
        assert kwargs["response_model"] is FeedbackAssessment
        return FeedbackAssessment(is_feedback=True, feedback_type="issue", content="Crack found")


def test_feedback_binds_the_current_action_without_model_ids() -> None:
    actor = ObservedActor(id="actor", name="operator")
    action = ObservedAction(id="action", start_time_ms=0, end_time_ms=1, actor_id=actor.id, action="Report crack")
    instruction = Instruction(id="i", name="i", segments=[Segment(id="s", scene="", actions=[action], actors=[actor], objects=[], uncertainties=[])])
    resolver = FeedbackResolver(instruction, extractor=Extractor())
    assert resolver.feedback_action_ids == ("action",)
    assert resolver.diagnostics[0].action_id == "action"
