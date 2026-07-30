from backend.extraction.sequnecer import Sequencer
from backend.schemas.process_knowledge.entities import Procedure
from backend.schemas.process_knowledge.text import (
    Instruction,
    ObservedAction,
    ObservedActor,
    Segment,
)


def make_action(action_id: str, name: str) -> ObservedAction:
    return ObservedAction(
        id=action_id,
        start_time_ms=0,
        end_time_ms=1000,
        actor_id=f"{action_id}-operator",
        action=name,
    )


def make_segment(
    segment_id: str,
    actions: list[ObservedAction],
) -> Segment:
    return Segment(
        id=segment_id,
        scene="workbench",
        actions=actions,
        actors=[
            ObservedActor(id=action.actor_id, name="operator") for action in actions
        ],
        objects=[],
        uncertainties=[],
    )


def test_sequencer_builds_procedure_steps_and_step_orders() -> None:
    instruction = Instruction(
        id="instruction-1",
        name="Assemble the frame",
        segments=[
            make_segment(
                "segment-1",
                [
                    make_action("action-1", "Align the parts"),
                    make_action("action-2", "Clamp the parts"),
                ],
            ),
            make_segment(
                "segment-2",
                [make_action("action-3", "Fasten the parts")],
            ),
        ],
    )

    sequencer = Sequencer(instruction)

    assert sequencer.procedure == Procedure(
        name="Assemble the frame",
        procedure_id="instruction-1",
    )
    assert [step.name for step in sequencer.steps] == [
        "Align the parts",
        "Clamp the parts",
        "Fasten the parts",
    ]
    assert [step.step_number for step in sequencer.steps] == [1, 2, 3]
    assert sequencer.steps_by_action_id["action-2"] == sequencer.steps[1]
    assert [
        (order.before.step_number, order.after.step_number)
        for order in sequencer.step_orders
    ] == [(1, 2), (2, 3)]


def test_sequencer_handles_instruction_without_actions() -> None:
    instruction = Instruction(
        id="instruction-1",
        name="Empty instruction",
        segments=[make_segment("segment-1", [])],
    )

    sequencer = Sequencer(instruction)

    assert sequencer.steps == []
    assert sequencer.steps_by_action_id == {}
    assert sequencer.step_orders == []
