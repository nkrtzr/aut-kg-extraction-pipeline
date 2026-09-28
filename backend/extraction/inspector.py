"""Create executions and extract action-scoped quality checks."""

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel

from backend.extraction.extractor import Extractor
from backend.extraction.sequnecer import Sequencer
from backend.schemas.process_knowledge.entities import QualityRequirement, Worker
from backend.schemas.process_knowledge.relations import QualityCheck, StepExecution
from backend.schemas.process_knowledge.text import ActionId, ObservedAction


class InspectionAssessment(BaseModel):
    """Quality-check assessment for the action supplied to the model."""

    is_quality_check: bool = False
    requirement: QualityRequirement | None = None
    result: Literal["pass", "fail", "conditional", "not_inspected"] | None = None
    findings: str | None = None
    corrective_action_required: bool = False
    corrective_action_description: str | None = None


class Inspector:
    """Build executions and bind quality checks to their known source action."""

    def __init__(self, sequencer: Sequencer, worker_bindings: Iterable[tuple[object, Worker]], *, extractor: Extractor | None = None) -> None:
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
        return tuple(action_id for action_id, execution in self.executions_by_action_id.items() if execution.step.step_type == "inspection")

    @staticmethod
    def _bind_workers(worker_bindings: Iterable[tuple[object, Worker]]) -> Mapping[ActionId, Worker]:
        bindings: dict[ActionId, Worker] = {}
        for observed_actor, worker in worker_bindings:
            for action_id in getattr(observed_actor, "action_ids", ()):
                bindings[action_id] = worker
        return bindings

    def _create_step_executions(self) -> None:
        for action in self.sequencer.actions:
            execution = StepExecution(
                step=self.sequencer.steps_by_action_id[action.id],
                executed_by=self._workers_by_action_id[action.id],
                actual_duration_seconds=(action.end_time_ms - action.start_time_ms) // 1000,
                source_text=action.model_dump_json(),
            )
            self.step_executions.append(execution)
            self.executions_by_action_id[action.id] = execution

    def _extract_quality_checks(self, extractor: Extractor) -> None:
        for action in self.sequencer.actions:
            assessment = extractor.extract(
                text=action.model_dump_json(),
                response_model=InspectionAssessment,
                system_prompt=self._quality_check_prompt(action),
            )
            if assessment.is_quality_check and assessment.requirement is not None:
                self._add_inspection(action.id, assessment)

    @staticmethod
    def _quality_check_prompt(action: ObservedAction) -> str:
        return (
            "Assess this one observed action for an explicit quality requirement "
            "or acceptance criterion. Return is_quality_check=false unless one is "
            "explicitly supported. If true, provide the requirement and supported "
            "result/finding fields. Do not return action IDs or references. "
            f"Current action text={action.action!r}."
        )

    def _add_inspection(self, action_id: ActionId, assessment: InspectionAssessment) -> None:
        requirement = assessment.requirement
        if requirement is None:
            return
        execution = self.executions_by_action_id[action_id]
        execution.step.step_type = "inspection"
        self.quality_requirements.append(requirement)
        self.quality_checks.append(QualityCheck(inspector=execution.executed_by, step_execution=execution, requirement=requirement, result=assessment.result, findings=assessment.findings, corrective_action_required=assessment.corrective_action_required, corrective_action_description=assessment.corrective_action_description))
