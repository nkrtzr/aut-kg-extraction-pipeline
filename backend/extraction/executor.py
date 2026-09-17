"""Assemble one procedure-execution record from completed pipeline stages."""

from collections.abc import Iterable

from backend.schemas.process_knowledge.entities import Procedure, Worker
from backend.schemas.process_knowledge.relations import (
    ProcedureExecution,
    QualityCheck,
    StepExecution,
    StepFeedback,
)


class ProcedureExecutionAssembler:
    """Create the top-level execution aggregate without inferring missing facts.

    Observed action timestamps are relative milliseconds, not absolute datetimes,
    so ``started_at`` and ``ended_at`` remain unset. Likewise, supervisor,
    status, and result are left unset until a source explicitly provides them.
    """

    def __init__(
        self,
        procedure: Procedure,
        step_executions: Iterable[StepExecution],
        *,
        quality_checks: Iterable[QualityCheck] = (),
        feedback: Iterable[StepFeedback] = (),
    ) -> None:
        self.step_executions = list(step_executions)
        self.quality_checks = list(quality_checks)
        self.feedback = list(feedback)
        self.procedure_execution = ProcedureExecution(
            procedure=procedure,
            executed_by=self._participating_workers(self.step_executions),
            step_executions=self.step_executions,
            quality_checks=self.quality_checks,
            feedback=self.feedback,
        )

    @staticmethod
    def _participating_workers(
        step_executions: Iterable[StepExecution],
    ) -> list[Worker]:
        """Return step executors once each, preserving execution order."""

        workers: list[Worker] = []
        for execution in step_executions:
            if not any(worker is execution.executed_by for worker in workers):
                workers.append(execution.executed_by)
        return workers
