"""Tests for the segmentation module."""

from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel

from backend.preprocessing.segmenter import (
    ExtractedAction,
    ExtractedObject,
    SegmentDraft,
    Segmenter,
    TextInstructionDraft,
)


class FakeExtractor:
    """Return prepared models without calling an LLM."""

    def __init__(self, results: list[BaseModel]):
        self.results: Iterator[BaseModel] = iter(results)
        self.calls: list[dict[str, object]] = []

    def extract(self, **kwargs: object) -> Any:
        self.calls.append(kwargs)
        return next(self.results)


def make_video_actions() -> list[ExtractedAction]:
    """Create two timestamped actions for a video example."""

    return [
        ExtractedAction(
            start_time="00:01:12.400",
            end_time="00:01:14.100",
            actor="worker_1",
            action="pick up",
            object_name="screwdriver",
        ),
        ExtractedAction(
            start_time="00:01:14.200",
            end_time="00:01:18.900",
            actor="worker_1",
            action="tighten",
            object_name="screw",
            instrument_name="screwdriver",
            target_name="metal panel",
        ),
    ]


def make_text_actions() -> list[ExtractedAction]:
    """Create two actions without timestamps for a text example."""

    return [
        ExtractedAction(
            actor="technician",
            action="pick up",
            object_name="wrench",
        ),
        ExtractedAction(
            actor="technician",
            action="tighten",
            object_name="bolt",
            instrument_name="wrench",
            target_name="frame",
        ),
    ]


def video_text() -> str:
    """Return one structured video-derived input."""

    return """
Video: indego_assembly_001
Instruction name: Indego assembly
Segment: seg_0003
Time: 00:01:12.400 - 00:01:18.900

Scene:
The worker picks up a screwdriver and tightens a screw
on the metal panel.

Actions:
1. [00:01:12.400 - 00:01:14.100]
   worker_1 picks up the screwdriver.

2. [00:01:14.200 - 00:01:18.900]
   worker_1 tightens the screw on the metal panel
   using the screwdriver.

Objects:
- screwdriver: tool
- screw: component
- metal panel: workpiece

Uncertainty:
- The exact screw type is not visible.
"""


def structured_text() -> str:
    """Return one structured text input without timestamps."""

    return """
Instruction ID: maintenance-guide
Instruction name: Tighten a frame bolt
Segment: text-segment-1

Scene:
A technician tightens a bolt on a frame.

Actions:
1. The technician picks up the wrench.
2. The technician tightens the bolt on the frame
   using the wrench.

Objects:
- technician: person
- wrench: tool
- bolt: component
- frame: workpiece

Uncertainty:
- The bolt size is not specified.
"""


def test_segmenter_creates_valid_video_instruction() -> None:
    fake_extractor = FakeExtractor(make_video_actions())
    segmenter = Segmenter(extractor=fake_extractor)

    instruction = segmenter.segment(video_text())

    assert instruction.id == "indego_assembly_001"
    assert instruction.name == "Indego assembly"
    assert len(instruction.segments) == 1

    segment = instruction.segments[0]

    assert segment.id == "seg_0003"
    assert segment.start_time_ms == 72_400
    assert segment.end_time_ms == 78_900
    assert len(segment.actions) == 2
    assert len(segment.actors) == 1
    assert len(segment.objects) == 3

    assert segment.actions[0].object_id == "seg_0003-object-1"
    assert segment.actions[1].object_id == "seg_0003-object-2"
    assert segment.actions[1].instrument_id == "seg_0003-object-1"
    assert segment.actions[1].target_id == "seg_0003-object-3"

    assert segment.actors[0].action_ids == [
        "seg_0003-action-1",
        "seg_0003-action-2",
    ]

    assert segment.objects[0].action_ids == [
        "seg_0003-action-1",
        "seg_0003-action-2",
    ]

    assert len(fake_extractor.calls) == 2

    assert all(
        call["response_model"] is ExtractedAction
        for call in fake_extractor.calls
    )


def test_video_source_can_be_selected_explicitly() -> None:
    fake_extractor = FakeExtractor(make_video_actions())
    segmenter = Segmenter(extractor=fake_extractor)

    instruction = segmenter.segment(
        video_text(),
        source_type="video",
        text_format="structured",
    )

    segment = instruction.segments[0]

    assert instruction.id == "indego_assembly_001"
    assert segment.start_time_ms == 72_400
    assert segment.actions[0].start_time_ms == 72_400
    assert segment.actions[0].end_time_ms == 74_100


def test_segmenter_rejects_empty_input() -> None:
    fake_extractor = FakeExtractor(make_video_actions())
    segmenter = Segmenter(extractor=fake_extractor)

    with pytest.raises(ValueError, match="must not be empty"):
        segmenter.segment("   ")

    assert fake_extractor.calls == []


def test_segmenter_rejects_missing_video_header() -> None:
    fake_extractor = FakeExtractor(make_video_actions())
    segmenter = Segmenter(extractor=fake_extractor)

    with pytest.raises(
        ValueError,
        match="Missing required header: Video",
    ):
        segmenter.segment(
            "Segment: seg_0003\n"
            "Time: 00:01:12.400 - 00:01:18.900"
        )

    assert fake_extractor.calls == []


def test_segmenter_allows_missing_uncertainty() -> None:
    fake_extractor = FakeExtractor(make_video_actions())
    segmenter = Segmenter(extractor=fake_extractor)

    text_without_uncertainty = video_text().split(
        "Uncertainty:",
        maxsplit=1,
    )[0]

    instruction = segmenter.segment(text_without_uncertainty)
    segment = instruction.segments[0]

    assert segment.uncertainties == []
    assert len(segment.actions) == 2
    assert len(fake_extractor.calls) == 2


def test_segmenter_keeps_two_actors_separate() -> None:
    actions = make_video_actions()
    actions[1].actor = "worker_2"

    fake_extractor = FakeExtractor(actions)
    segmenter = Segmenter(extractor=fake_extractor)

    instruction = segmenter.segment(video_text())
    segment = instruction.segments[0]

    assert len(segment.actors) == 2
    assert segment.actors[0].name == "worker_1"
    assert segment.actors[1].name == "worker_2"

    assert segment.actors[0].action_ids == [
        "seg_0003-action-1",
    ]

    assert segment.actors[1].action_ids == [
        "seg_0003-action-2",
    ]

    assert segment.actions[0].actor_id == "seg_0003-actor-1"
    assert segment.actions[1].actor_id == "seg_0003-actor-2"


def test_segmenter_rejects_invalid_timestamp() -> None:
    fake_extractor = FakeExtractor(make_video_actions())
    segmenter = Segmenter(extractor=fake_extractor)

    invalid_text = video_text().replace(
        "Time: 00:01:12.400 - 00:01:18.900",
        "Time: 00:61:12.400 - 00:01:18.900",
    )

    with pytest.raises(ValueError, match="Invalid timestamp"):
        segmenter.segment(invalid_text)


def test_segmenter_handles_four_actions() -> None:
    actions = make_video_actions()

    actions.extend(
        [
            ExtractedAction(
                start_time="00:01:19.000",
                end_time="00:01:21.000",
                actor="worker_1",
                action="place",
                object_name="screw",
                target_name="metal panel",
            ),
            ExtractedAction(
                start_time="00:01:21.200",
                end_time="00:01:24.000",
                actor="worker_1",
                action="inspect",
                object_name="metal panel",
            ),
        ]
    )

    extra_action_text = """
3. [00:01:19.000 - 00:01:21.000]
   worker_1 places the screw on the metal panel.

4. [00:01:21.200 - 00:01:24.000]
   worker_1 inspects the metal panel.
"""

    long_text = video_text().replace(
        "\nObjects:",
        f"\n{extra_action_text}\nObjects:",
    ).replace(
        "Time: 00:01:12.400 - 00:01:18.900",
        "Time: 00:01:12.400 - 00:01:24.000",
    )

    fake_extractor = FakeExtractor(actions)
    segmenter = Segmenter(extractor=fake_extractor)

    instruction = segmenter.segment(long_text)
    segment = instruction.segments[0]

    assert len(segment.actions) == 4
    assert len(fake_extractor.calls) == 4
    assert segment.actions[2].action == "place"
    assert segment.actions[3].action == "inspect"

    assert segment.actors[0].action_ids == [
        "seg_0003-action-1",
        "seg_0003-action-2",
        "seg_0003-action-3",
        "seg_0003-action-4",
    ]


def test_segmenter_corrects_inspection_object_role() -> None:
    actions = make_video_actions()

    actions[1] = ExtractedAction(
        start_time="00:01:14.200",
        end_time="00:01:18.900",
        actor="worker_1",
        action="inspect",
        object_name=None,
        instrument_name=None,
        target_name="metal panel",
    )

    fake_extractor = FakeExtractor(actions)
    segmenter = Segmenter(extractor=fake_extractor)

    instruction = segmenter.segment(video_text())
    action = instruction.segments[0].actions[1]

    assert action.object_id == "seg_0003-object-3"
    assert action.target_id is None


def test_segmenter_recovers_missing_target() -> None:
    actions = make_video_actions()
    actions[1].target_name = None

    fake_extractor = FakeExtractor(actions)
    segmenter = Segmenter(extractor=fake_extractor)

    instruction = segmenter.segment(video_text())
    action = instruction.segments[0].actions[1]

    assert action.object_id == "seg_0003-object-2"
    assert action.instrument_id == "seg_0003-object-1"
    assert action.target_id == "seg_0003-object-3"


def test_structured_text_creates_instruction_without_timestamps() -> None:
    fake_extractor = FakeExtractor(make_text_actions())
    segmenter = Segmenter(extractor=fake_extractor)

    instruction = segmenter.segment(
        structured_text(),
        source_type="text",
        text_format="structured",
    )

    assert instruction.id == "maintenance-guide"
    assert instruction.name == "Tighten a frame bolt"
    assert len(instruction.segments) == 1

    segment = instruction.segments[0]

    assert segment.id == "text-segment-1"
    assert segment.scene == "A technician tightens a bolt on a frame."
    assert len(segment.actions) == 2
    assert len(segment.actors) == 1
    assert len(segment.objects) == 4

    assert segment.actions[0].start_time_ms is None
    assert segment.actions[0].end_time_ms is None
    assert segment.actions[1].start_time_ms is None
    assert segment.actions[1].end_time_ms is None

    assert segment.actions[1].object_id == "text-segment-1-object-3"
    assert segment.actions[1].instrument_id == "text-segment-1-object-2"
    assert segment.actions[1].target_id == "text-segment-1-object-4"

    assert len(fake_extractor.calls) == 2


def test_structured_text_removes_extracted_timestamps() -> None:
    actions = make_text_actions()

    actions[0].start_time = "00:00:01.000"
    actions[0].end_time = "00:00:02.000"

    fake_extractor = FakeExtractor(actions)
    segmenter = Segmenter(extractor=fake_extractor)

    instruction = segmenter.segment(
        structured_text(),
        source_type="text",
        text_format="structured",
    )

    action = instruction.segments[0].actions[0]

    assert action.start_time_ms is None
    assert action.end_time_ms is None


def test_structured_text_accepts_explicit_instruction_metadata() -> None:
    fake_extractor = FakeExtractor(make_text_actions())
    segmenter = Segmenter(extractor=fake_extractor)

    instruction = segmenter.segment(
        structured_text(),
        source_type="text",
        text_format="structured",
        instruction_id="manual-001",
        instruction_name="Manual bolt procedure",
    )

    assert instruction.id == "manual-001"
    assert instruction.name == "Manual bolt procedure"


def test_free_form_text_is_normalized_into_segments() -> None:
    normalized = TextInstructionDraft(
        instruction_id="bearing-maintenance",
        instruction_name="Replace a bearing",
        segments=[
            SegmentDraft(
                segment_id="text-segment-1",
                scene="A technician removes the worn bearing.",
                actions=[
                    ExtractedAction(
                        actor="technician",
                        action="remove",
                        object_name="bearing",
                        instrument_name="puller",
                    )
                ],
                objects=[
                    ExtractedObject(
                        name="technician",
                        object_type="person",
                    ),
                    ExtractedObject(
                        name="bearing",
                        object_type="component",
                    ),
                    ExtractedObject(
                        name="puller",
                        object_type="tool",
                    ),
                ],
            ),
            SegmentDraft(
                segment_id="text-segment-2",
                scene="The technician installs a new bearing.",
                actions=[
                    ExtractedAction(
                        actor="technician",
                        action="install",
                        object_name="new bearing",
                        target_name="housing",
                    )
                ],
                objects=[
                    ExtractedObject(
                        name="technician",
                        object_type="person",
                    ),
                    ExtractedObject(
                        name="new bearing",
                        object_type="component",
                    ),
                    ExtractedObject(
                        name="housing",
                        object_type="workpiece",
                    ),
                ],
            ),
        ],
    )

    fake_extractor = FakeExtractor([normalized])
    segmenter = Segmenter(extractor=fake_extractor)

    instruction = segmenter.segment(
        (
            "First remove the worn bearing with a puller. "
            "Then install a new bearing in the housing."
        ),
        source_type="text",
        text_format="free_form",
    )

    assert instruction.id == "bearing-maintenance"
    assert instruction.name == "Replace a bearing"
    assert len(instruction.segments) == 2

    first_segment = instruction.segments[0]
    second_segment = instruction.segments[1]

    assert first_segment.id == "text-segment-1"
    assert second_segment.id == "text-segment-2"
    assert first_segment.actions[0].start_time_ms is None
    assert first_segment.actions[0].end_time_ms is None
    assert second_segment.actions[0].start_time_ms is None
    assert second_segment.actions[0].end_time_ms is None

    assert first_segment.actions[0].instrument_id == (
        "text-segment-1-object-3"
    )

    assert second_segment.actions[0].target_id == (
        "text-segment-2-object-3"
    )

    assert len(fake_extractor.calls) == 1
    assert fake_extractor.calls[0]["response_model"] is (
        TextInstructionDraft
    )


def test_free_form_text_removes_model_generated_timestamps() -> None:
    normalized = TextInstructionDraft(
        instruction_id="cleaning-guide",
        instruction_name="Clean a panel",
        segments=[
            SegmentDraft(
                segment_id="text-segment-1",
                scene="A worker wipes the panel.",
                start_time="00:00:01.000",
                end_time="00:00:05.000",
                actions=[
                    ExtractedAction(
                        actor="worker",
                        action="wipe",
                        object_name="panel",
                        start_time="00:00:01.000",
                        end_time="00:00:05.000",
                    )
                ],
                objects=[
                    ExtractedObject(
                        name="worker",
                        object_type="person",
                    ),
                    ExtractedObject(
                        name="panel",
                        object_type="workpiece",
                    ),
                ],
            )
        ],
    )

    fake_extractor = FakeExtractor([normalized])
    segmenter = Segmenter(extractor=fake_extractor)

    instruction = segmenter.segment(
        "A worker wipes the panel.",
        source_type="text",
        text_format="free_form",
    )

    action = instruction.segments[0].actions[0]

    assert action.start_time_ms is None
    assert action.end_time_ms is None


def test_free_form_text_accepts_explicit_instruction_metadata() -> None:
    normalized = TextInstructionDraft(
        instruction_id="generated-id",
        instruction_name="Generated name",
        segments=[
            SegmentDraft(
                segment_id="text-segment-1",
                scene="A worker checks a panel.",
                actions=[
                    ExtractedAction(
                        actor="worker",
                        action="check",
                        object_name="panel",
                    )
                ],
                objects=[
                    ExtractedObject(
                        name="worker",
                        object_type="person",
                    ),
                    ExtractedObject(
                        name="panel",
                        object_type="workpiece",
                    ),
                ],
            )
        ],
    )

    fake_extractor = FakeExtractor([normalized])
    segmenter = Segmenter(extractor=fake_extractor)

    instruction = segmenter.segment(
        "Check the panel.",
        source_type="text",
        text_format="free_form",
        instruction_id="manual-id",
        instruction_name="Manual name",
    )

    assert instruction.id == "manual-id"
    assert instruction.name == "Manual name"


def test_free_form_text_rejects_empty_normalized_result() -> None:
    normalized = TextInstructionDraft(
        instruction_id="empty-instruction",
        instruction_name="Empty instruction",
        segments=[],
    )

    fake_extractor = FakeExtractor([normalized])
    segmenter = Segmenter(extractor=fake_extractor)

    with pytest.raises(
        ValueError,
        match="did not produce any segments",
    ):
        segmenter.segment(
            "This input contains no process step.",
            source_type="text",
            text_format="free_form",
        )


def test_video_action_requires_timestamps() -> None:
    actions = make_video_actions()
    actions[0].start_time = None

    fake_extractor = FakeExtractor(actions)
    segmenter = Segmenter(extractor=fake_extractor)

    with pytest.raises(
        ValueError,
        match="must contain start and end timestamps",
    ):
        segmenter.segment(
            video_text(),
            source_type="video",
        )


def test_video_rejects_free_form_format() -> None:
    fake_extractor = FakeExtractor([])
    segmenter = Segmenter(extractor=fake_extractor)

    with pytest.raises(
        ValueError,
        match="Video input must use the structured format",
    ):
        segmenter.segment(
            video_text(),
            source_type="video",
            text_format="free_form",
        )

    assert fake_extractor.calls == []


def test_segmenter_rejects_unknown_source_type() -> None:
    fake_extractor = FakeExtractor([])
    segmenter = Segmenter(extractor=fake_extractor)

    with pytest.raises(
        ValueError,
        match="source_type must be 'video' or 'text'",
    ):
        segmenter.segment(
            structured_text(),
            source_type="audio",  # type: ignore[arg-type]
        )

    assert fake_extractor.calls == []


def test_segmenter_rejects_unknown_text_format() -> None:
    fake_extractor = FakeExtractor([])
    segmenter = Segmenter(extractor=fake_extractor)

    with pytest.raises(
        ValueError,
        match="text_format must be 'structured' or 'free_form'",
    ):
        segmenter.segment(
            structured_text(),
            source_type="text",
            text_format="automatic",  # type: ignore[arg-type]
        )

    assert fake_extractor.calls == []