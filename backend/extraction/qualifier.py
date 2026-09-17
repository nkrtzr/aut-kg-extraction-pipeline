"""Extract explicit worker qualification records from relevant segments."""

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

from backend.extraction.extractor import Extractor
from backend.schemas.process_knowledge.entities import Worker
from backend.schemas.process_knowledge.relations import QualificationRecord
from backend.schemas.process_knowledge.text import Instruction, Segment


class ExtractedQualification(BaseModel):
    """Qualification data before it is bound to a known worker."""

    expertise_level: Literal["apprentice", "skilled", "expert", "master"]
    procedure_domain: str
    certifying_body: str | None = None
    certification_norm: str | None = None
    certificate_id: str | None = None


class WorkerQualifications(BaseModel):
    """Structured qualification extraction response for one known worker."""

    qualifications: list[ExtractedQualification] = Field(default_factory=list)


class QualificationResolver:
    """Extract explicit qualifications for known workers.

    This stage does not create workers.  Each LLM request is scoped to one
    already-resolved worker and the first relevant segments in instruction
    order, preventing qualifications from being transferred between workers.
    """

    def __init__(
        self,
        instruction: Instruction,
        worker_bindings: Iterable[tuple[object, Worker]],
        *,
        extractor: Extractor | None = None,
        segments_per_worker: int | None = None,
    ) -> None:
        if segments_per_worker is not None and segments_per_worker < 1:
            raise ValueError("segments_per_worker must be at least one or None")

        self.instruction = instruction
        self.segments_per_worker = segments_per_worker
        self.qualification_records: list[QualificationRecord] = []
        self._extract_qualifications(
            worker_bindings,
            extractor or Extractor(),
        )

    def _extract_qualifications(
        self,
        worker_bindings: Iterable[tuple[object, Worker]],
        extractor: Extractor,
    ) -> None:
        for observed_actor, worker in worker_bindings:
            segments = self._relevant_segments(observed_actor)
            if not segments:
                continue
            result = extractor.extract(
                text=self._source_text(worker, segments),
                response_model=WorkerQualifications,
                system_prompt=self._qualification_prompt(worker),
            )
            self.qualification_records.extend(
                QualificationRecord(
                    worker=worker,
                    **qualification.model_dump(),
                )
                for qualification in result.qualifications
            )

    def _relevant_segments(self, observed_actor: object) -> list[Segment]:
        action_ids = set(getattr(observed_actor, "action_ids", ()))
        relevant = [
            segment
            for segment in self.instruction.segments
            if any(action.id in action_ids for action in segment.actions)
        ]
        if self.segments_per_worker is not None:
            return relevant[: self.segments_per_worker]
        return relevant

    @staticmethod
    def _source_text(worker: Worker, segments: list[Segment]) -> str:
        return WorkerQualificationSource(
            worker_name=worker.name,
            segments=segments,
        ).model_dump_json()

    def _qualification_prompt(self, worker: Worker) -> str:
        return (
            "Extract qualification records only for the supplied known worker, "
            f"{worker.name!r}. Return a record only when the supplied segments "
            "explicitly support an expertise level, procedure domain, training, "
            "certificate, certification norm, or certifying body. Do not infer "
            "qualifications from a job title, generic participation, or other "
            "workers' evidence. Each record must use one of the supported "
            "expertise levels and a concrete procedure_domain. If there is no "
            "explicit qualification evidence, return an empty qualifications list. "
            f"The surrounding instruction is id={self.instruction.id!r}, "
            f"name={self.instruction.name!r}."
        )


class WorkerQualificationSource(BaseModel):
    """The intentionally limited source context sent for one worker."""

    worker_name: str
    segments: list[Segment]
