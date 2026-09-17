"""Create execution records and extract quality checks from observations.

An observed action always remains a :class:`Step`: it represents the ordered
procedure specification.  This stage adds the corresponding
``StepExecution`` occurrence.  Inspection classification is deliberately kept
separate, so an inspection step is never removed or replaced by a
``QualityCheck``.
"""

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, Field

from backend.extraction.extractor import Extractor
from backend.extraction.sequnecer import Sequencer
from backend.schemas.process_knowledge.entities import QualityRequirement, Worker
from backend.schemas.process_knowledge.relations import QualityCheck, StepExecution
from backend.schemas.process_knowledge.text import ActionId, Segment


class ActionInspection(BaseModel):
    """A quality check explicitly supported by one observed action."""

    action_id: ActionId
    requirement: QualityRequirement
    result: Literal["pass", "fail", "conditional", "not_inspected"] | None = None
    findings: str | None = None
    corrective_action_required: bool = False
    corrective_action_description: str | None = None


class SegmentInspection(BaseModel):
    """Structured quality-check extraction result for one segment."""

    actions: list[ActionInspection] = Field(default_factory=list)


class Inspector:
    """Build one execution for each observed action.

    The LLM returns only actions that have an explicit, supported quality
    requirement. Those actions are marked as inspection steps and receive a
    ``QualityCheck``. Other actions still receive a ``StepExecution`` but are
    not treated as quality checks.
    """

    def __init__(
        self,
        sequencer: Sequencer,
        worker_bindings: Iterable[tuple[object, Worker]],
        *,
        extractor: Extractor | None = None,
    ) -> None:
        self.sequencer = sequencer
        self._workers_by_action_id = self._bind_workers(worker_bindings)
        self.step_executions: list[StepExecution] = []
        self.executions_by_action_id: dict[ActionId, StepExecution] = {}
        self.quality_requirements: list[QualityRequirement] = []
        self.quality_checks: list[QualityCheck] = []
        self._create_step_executions()
        self._extract_quality_checks(extractor or Extractor())

    @property
    def inspection_action_ids(self) -> tuple[ActionId, ...]:
        """Action IDs whose LLM-extracted requirement makes them inspections."""

        return tuple(
            action_id
            for action_id, execution in self.executions_by_action_id.items()
            if execution.step.step_type == "inspection"
        )

    @staticmethod
    def _bind_workers(
        worker_bindings: Iterable[tuple[object, Worker]],
    ) -> Mapping[ActionId, Worker]:
        workers_by_action_id: dict[ActionId, Worker] = {}
        for observed_actor, worker in worker_bindings:
            action_ids = getattr(observed_actor, "action_ids", ())
            for action_id in action_ids:
                existing = workers_by_action_id.get(action_id)
                if existing is not None and existing is not worker:
                    raise ValueError(
                        f"Action {action_id!r} resolves to multiple workers"
                    )
                workers_by_action_id[action_id] = worker
        return workers_by_action_id

    def _create_step_executions(self) -> None:
        for action in self.sequencer.actions:
            step = self.sequencer.steps_by_action_id[action.id]
            try:
                worker = self._workers_by_action_id[action.id]
            except KeyError as error:
                raise ValueError(
                    f"Action {action.id!r} has no resolved worker"
                ) from error

            execution = StepExecution(
                step=step,
                executed_by=worker,
                actual_duration_seconds=(
                    action.end_time_ms - action.start_time_ms
                ) // 1000,
                source_text=action.model_dump_json(),
            )
            self.step_executions.append(execution)
            self.executions_by_action_id[action.id] = execution

    def _extract_quality_checks(self, extractor: Extractor) -> None:
        for segment in self.sequencer.instruction.segments:
            inspection = extractor.extract(
                text=segment.model_dump_json(),
                response_model=SegmentInspection,
                system_prompt=self._quality_check_prompt(segment),
            )
            self._add_segment_inspection(segment, inspection)

    def _quality_check_prompt(self, segment: Segment) -> str:
        action_ids = [action.id for action in segment.actions]
        return (
            "Analyze this industrial instruction segment for quality checks. "
            "Return an ActionInspection only when an action explicitly states "
            "or provides clear evidence for a quality requirement or acceptance "
            "criterion. Associate it with the exact action_id and create a "
            "QualityRequirement describing that criterion. Do not classify a "
            "generic process action as a quality check, do not invent criteria, "
            "and omit actions without a supported requirement. If the text "
            "states a result, finding, or corrective action, capture it; otherwise "
            "leave those fields null or false. Return only action IDs from: "
            f"{action_ids!r}. The surrounding instruction is "
            f"id={self.sequencer.instruction.id!r}, "
            f"name={self.sequencer.instruction.name!r}."
        )

    def _add_segment_inspection(
        self,
        segment: Segment,
        inspection: SegmentInspection,
    ) -> None:
        valid_action_ids = {action.id for action in segment.actions}
        returned_action_ids = [item.action_id for item in inspection.actions]
        if len(returned_action_ids) != len(set(returned_action_ids)):
            raise ValueError("Segment inspection returned a duplicate action ID")
        if not set(returned_action_ids).issubset(valid_action_ids):
            raise ValueError(
                "Segment inspection returned an action ID outside its segment"
            )

        for item in inspection.actions:
            execution = self.executions_by_action_id[item.action_id]
            execution.step.step_type = "inspection"
            self.quality_requirements.append(item.requirement)
            self.quality_checks.append(
                QualityCheck(
                    inspector=execution.executed_by,
                    step_execution=execution,
                    requirement=item.requirement,
                    result=item.result,
                    findings=item.findings,
                    corrective_action_required=item.corrective_action_required,
                    corrective_action_description=item.corrective_action_description,
                )
            )
