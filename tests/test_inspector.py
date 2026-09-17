import pytest

from typing import Any

from backend.extraction.inspector import (
    ActionInspection,
    Inspector,
    SegmentInspection,
)
from backend.extraction.sequnecer import Sequencer
from backend.schemas.process_knowledge.entities import QualityRequirement, Worker
from backend.schemas.process_knowledge.text import (
    Instruction,
    ObservedAction,
    ObservedActor,
    Segment,
)


def make_instruction() -> tuple[Instruction, ObservedActor]:
    actor = ObservedActor(id="worker-1", name="Inspector")
    instruction = Instruction(
        id="instruction-1",
        name="Inspect assembly",
        segments=[
            Segment(
                id="segment-1",
                scene="bench",
                actions=[
                    ObservedAction(
                        id="action-1",
                        start_time_ms=0,
                        end_time_ms=2500,
                        actor_id=actor.id,
                        action="Assemble the bracket",
                    ),
                    ObservedAction(
                        id="action-2",
                        start_time_ms=2500,
                        end_time_ms=5000,
                        actor_id=actor.id,
                        action="Inspect the finished bracket",
                    ),
                ],
                actors=[actor],
                objects=[],
                uncertainties=[],
            )
        ],
    )
    return instruction, actor


class InspectionExtractor:
    def extract(self, **kwargs: Any) -> SegmentInspection:
        assert kwargs["response_model"] is SegmentInspection
        return SegmentInspection(
            actions=[
                ActionInspection(
                    action_id="action-2",
                    requirement=QualityRequirement(
                        name="Finished bracket is free from defects",
                        requirement_type="visual",
                    ),
                    result="pass",
                )
            ]
        )


def test_inspector_preserves_steps_and_marks_quality_actions() -> None:
    instruction, actor = make_instruction()
    sequencer = Sequencer(instruction)
    worker = Worker(name=actor.name)

    inspector = Inspector(
        sequencer,
        [(actor, worker)],
        extractor=InspectionExtractor(),
    )

    assert len(inspector.step_executions) == 2
    assert inspector.step_executions[1].step is sequencer.steps[1]
    assert inspector.step_executions[1].executed_by is worker
    assert inspector.step_executions[0].actual_duration_seconds == 2
    assert inspector.inspection_action_ids == ("action-2",)
    assert inspector.step_executions[1].step.step_type == "inspection"
    assert inspector.quality_checks[0].requirement is inspector.quality_requirements[0]
    assert inspector.quality_checks[0].result == "pass"


def test_inspector_rejects_actions_without_workers() -> None:
    instruction, _ = make_instruction()

    with pytest.raises(ValueError, match="has no resolved worker"):
        Inspector(Sequencer(instruction), [], extractor=InspectionExtractor())
