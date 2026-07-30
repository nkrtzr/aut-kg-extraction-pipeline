import pytest

from backend.schemas.process_knowledge.text import (
    Instruction,
    ObservedAction,
    ObservedActor,
    ObservedObject,
    Segment,
    VideoSegment,
)


def make_video_segment(
    objects: list[ObservedObject],
    actors: list[ObservedActor] | None = None,
) -> VideoSegment:
    return VideoSegment(
        id="segment-1",
        start_time_ms=0,
        end_time_ms=1000,
        scene="workbench",
        actions=[],
        actors=actors or [],
        objects=objects,
        uncertainties=[],
    )


def test_video_segment_accepts_numbered_repeated_objects() -> None:
    segment = make_video_segment(
        [
            ObservedObject(id="cup-1", name="cup_1", object_type="cup"),
            ObservedObject(id="cup-2", name="cup_2", object_type="cup"),
        ]
    )

    assert [obj.name for obj in segment.objects] == ["cup_1", "cup_2"]


def test_video_segment_numbers_repeated_objects_without_changing_ids() -> None:
    segment = make_video_segment(
        [
            ObservedObject(id="cup-1", name="cup", object_type="cup"),
            ObservedObject(id="cup-2", name="cup", object_type="cup"),
        ]
    )

    assert [obj.name for obj in segment.objects] == ["cup_1", "cup_2"]
    assert [obj.id for obj in segment.objects] == ["cup-1", "cup-2"]


def test_video_segment_numbers_repeated_actors_without_changing_ids() -> None:
    segment = make_video_segment(
        objects=[],
        actors=[
            ObservedActor(id="worker-1", name="worker"),
            ObservedActor(id="worker-2", name="worker"),
        ],
    )

    assert [actor.name for actor in segment.actors] == ["worker_1", "worker_2"]
    assert [actor.id for actor in segment.actors] == ["worker-1", "worker-2"]


def test_instruction_indexes_typed_references_and_reverse_references() -> None:
    segment = Segment(
        id="segment-1",
        scene="workbench",
        actions=[
            ObservedAction(
                id="action-1",
                start_time_ms=0,
                end_time_ms=1,
                actor_id="worker-1",
                action="strike",
                object_id="nail-1",
                instrument_id="hammer-1",
            ),
            ObservedAction(
                id="action-2",
                start_time_ms=1,
                end_time_ms=2,
                actor_id="worker-1",
                action="inspect",
                object_id="nail-1",
            ),
        ],
        actors=[ObservedActor(id="worker-1", name="worker")],
        objects=[
            ObservedObject(id="hammer-1", name="hammer", object_type="tool"),
            ObservedObject(id="nail-1", name="nail", object_type="fastener"),
        ],
        uncertainties=[],
    )

    instruction = Instruction(
        id="instruction-1",
        name="Drive a nail",
        segments=[segment],
    )

    index = instruction.reference_index
    assert index.actions_by_id["action-1"] == segment.actions[0]
    assert index.actors_by_id["worker-1"] == segment.actors[0]
    assert index.objects_by_id["hammer-1"] == segment.objects[0]
    assert index.actor_action_ids["worker-1"] == ("action-1", "action-2")
    assert index.object_action_ids["hammer-1"] == ("action-1",)
    assert index.object_action_ids["nail-1"] == ("action-1", "action-2")


def test_instruction_rejects_duplicate_segment_ids() -> None:
    segment = Segment(
        id="segment-1",
        scene="workbench",
        actions=[],
        actors=[],
        objects=[],
        uncertainties=[],
    )

    with pytest.raises(ValueError, match="segment IDs must be unique"):
        Instruction(
            id="instruction-1",
            name="Duplicate segments",
            segments=[segment, segment.model_copy(deep=True)],
        )


def test_instruction_rejects_duplicate_action_ids_across_segments() -> None:
    def segment(segment_id: str, actor_id: str) -> Segment:
        return Segment(
            id=segment_id,
            scene="workbench",
            actions=[
                ObservedAction(
                    id="action-1",
                    start_time_ms=0,
                    end_time_ms=1000,
                    actor_id=actor_id,
                    action="inspect",
                ),
            ],
            actors=[ObservedActor(id=actor_id, name="worker")],
            objects=[],
            uncertainties=[],
        )

    with pytest.raises(ValueError, match="action IDs must be globally unique"):
        Instruction(
            id="instruction-1",
            name="Duplicate actions",
            segments=[
                segment("segment-1", "actor-1"),
                segment("segment-2", "actor-2"),
            ],
        )


def test_instruction_rejects_unknown_actor_endpoint() -> None:
    segment = Segment(
        id="segment-1",
        scene="workbench",
        actions=[
            ObservedAction(
                id="action-1",
                start_time_ms=0,
                end_time_ms=1,
                actor_id="missing-actor",
                action="inspect",
            ),
        ],
        actors=[],
        objects=[],
        uncertainties=[],
    )

    with pytest.raises(ValueError, match="references unknown actor"):
        Instruction(id="instruction-1", name="Inspect", segments=[segment])


def test_instruction_rejects_cross_segment_object_endpoint() -> None:
    segment_1 = Segment(
        id="segment-1",
        scene="workbench",
        actions=[
            ObservedAction(
                id="action-1",
                start_time_ms=0,
                end_time_ms=1,
                actor_id="actor-1",
                action="strike",
                instrument_id="hammer-2",
            ),
        ],
        actors=[ObservedActor(id="actor-1", name="worker")],
        objects=[],
        uncertainties=[],
    )
    segment_2 = Segment(
        id="segment-2",
        scene="workbench",
        actions=[],
        actors=[],
        objects=[
            ObservedObject(id="hammer-2", name="hammer", object_type="tool"),
        ],
        uncertainties=[],
    )

    with pytest.raises(ValueError, match="references unknown objects"):
        Instruction(
            id="instruction-1",
            name="Strike",
            segments=[segment_1, segment_2],
        )


def test_instruction_reference_maps_are_read_only() -> None:
    instruction = Instruction(
        id="instruction-1",
        name="Empty",
        segments=[],
    )

    with pytest.raises(TypeError):
        instruction.reference_index.actions_by_id["action-1"] = ObservedAction(
            id="action-1",
            start_time_ms=0,
            end_time_ms=1,
            actor_id="actor-1",
            action="inspect",
        )
