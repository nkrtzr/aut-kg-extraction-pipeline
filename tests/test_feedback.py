from typing import Any

from backend.extraction.feedback import ActionFeedback, FeedbackResolver, SegmentFeedback
from backend.extraction.inspector import Inspector, SegmentInspection
from backend.extraction.sequnecer import Sequencer
from backend.schemas.process_knowledge.entities import Worker
from backend.schemas.process_knowledge.text import (
    Instruction,
    ObservedAction,
    ObservedActor,
    Segment,
)


class FeedbackExtractor:
    def extract(self, **kwargs: Any) -> Any:
        if kwargs["response_model"] is SegmentFeedback:
            return SegmentFeedback(
                feedback=[
                    ActionFeedback(
                        feedback_action_id="action-2",
                        step_action_id="action-1",
                        feedback_type="issue",
                        content="The bracket has a visible crack.",
                        severity="critical",
                    )
                ]
            )
        assert kwargs["response_model"] is SegmentInspection
        return SegmentInspection()


def test_feedback_actions_are_excluded_from_steps_and_linked_to_execution() -> None:
    actor = ObservedActor(id="actor-1", name="operator")
    instruction = Instruction(
        id="instruction-1",
        name="Assemble bracket",
        segments=[
            Segment(
                id="segment-1",
                scene="bench",
                actions=[
                    ObservedAction(
                        id="action-1",
                        start_time_ms=0,
                        end_time_ms=1000,
                        actor_id=actor.id,
                        action="Assemble the bracket",
                    ),
                    ObservedAction(
                        id="action-2",
                        start_time_ms=1000,
                        end_time_ms=2000,
                        actor_id=actor.id,
                        action="Report a visible crack in the bracket",
                    ),
                ],
                actors=[actor],
                objects=[],
                uncertainties=[],
            )
        ],
    )
    worker = Worker(name="operator")
    extractor = FeedbackExtractor()
    resolver = FeedbackResolver(instruction, [(actor, worker)], extractor=extractor)

    sequencer = Sequencer(
        instruction,
        excluded_action_ids=set(resolver.feedback_action_ids),
    )
    inspector = Inspector(sequencer, [(actor, worker)], extractor=extractor)
    feedback = resolver.create_step_feedback(inspector.executions_by_action_id)

    assert tuple(sequencer.steps_by_action_id) == ("action-1",)
    assert len(feedback) == 1
    assert feedback[0].step_execution is inspector.executions_by_action_id["action-1"]
    assert feedback[0].reported_by is worker
    assert feedback[0].feedback_type == "issue"
