from typing import Any

import pytest

from backend.extraction.qualifier import (
    ExtractedQualification,
    QualificationResolver,
    WorkerQualifications,
)
from backend.schemas.process_knowledge.entities import Worker
from backend.schemas.process_knowledge.text import (
    Instruction,
    ObservedAction,
    ObservedActor,
    Segment,
)


class QualificationExtractor:
    def __init__(self) -> None:
        self.source_texts: list[str] = []

    def extract(self, **kwargs: Any) -> WorkerQualifications:
        assert kwargs["response_model"] is WorkerQualifications
        self.source_texts.append(kwargs["text"])
        return WorkerQualifications(
            qualifications=[
                ExtractedQualification(
                    expertise_level="skilled",
                    procedure_domain="welding",
                    certification_norm="ISO 9606",
                )
            ]
        )


def make_instruction() -> tuple[Instruction, ObservedActor]:
    actor = ObservedActor(id="actor-1", name="Alex")
    segments = []
    for number in range(1, 4):
        segment_actor = ObservedActor(id=f"actor-{number}", name="Alex")
        segments.append(
            Segment(
                id=f"segment-{number}",
                scene="workshop",
                actions=[
                    ObservedAction(
                        id=f"action-{number}",
                        start_time_ms=0,
                        end_time_ms=1000,
                        actor_id=segment_actor.id,
                        action="Weld component",
                    )
                ],
                actors=[segment_actor],
                objects=[],
                uncertainties=[],
            )
        )
    return Instruction(id="instruction-1", name="Welding", segments=segments), actor


def test_qualification_resolver_limits_context_per_worker() -> None:
    instruction, actor = make_instruction()
    actor.action_ids = ["action-1", "action-2", "action-3"]
    extractor = QualificationExtractor()

    resolver = QualificationResolver(
        instruction,
        [(actor, Worker(name="Alex"))],
        extractor=extractor,
        segments_per_worker=2,
    )

    assert len(resolver.qualification_records) == 1
    record = resolver.qualification_records[0]
    assert record.worker.name == "Alex"
    assert record.procedure_domain == "welding"
    assert "segment-1" in extractor.source_texts[0]
    assert "segment-2" in extractor.source_texts[0]
    assert "segment-3" not in extractor.source_texts[0]


def test_qualification_resolver_rejects_non_positive_segment_limit() -> None:
    instruction, actor = make_instruction()

    with pytest.raises(ValueError, match="at least one"):
        QualificationResolver(
            instruction,
            [(actor, Worker(name="Alex"))],
            segments_per_worker=0,
        )
