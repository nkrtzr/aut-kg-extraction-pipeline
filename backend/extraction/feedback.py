"""Identify feedback actions before sequencing and attach them to executions."""

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, Field

from backend.extraction.extractor import Extractor
from backend.schemas.process_knowledge.entities import Worker
from backend.schemas.process_knowledge.relations import StepExecution, StepFeedback
from backend.schemas.process_knowledge.text import ActionId, Instruction, Segment


class ActionFeedback(BaseModel):
    """Feedback reported by one observed action about a procedural action."""

    feedback_action_id: ActionId
    step_action_id: ActionId
    feedback_type: Literal["question", "issue", "suggestion", "observation"]
    content: str
    severity: Literal["info", "warning", "critical"] | None = None
    resolved: bool = False
    resolution: str | None = None


class SegmentFeedback(BaseModel):
    """Structured feedback-classification result for one segment."""

    feedback: list[ActionFeedback] = Field(default_factory=list)


class FeedbackResolver:
    """Classify explicit feedback actions before they become procedural steps."""

    def __init__(
        self,
        instruction: Instruction,
        worker_bindings: Iterable[tuple[object, Worker]],
        *,
        extractor: Extractor | None = None,
    ) -> None:
        self.instruction = instruction
        self._workers_by_action_id = self._bind_workers(worker_bindings)
        self.feedback_by_action_id: dict[ActionId, ActionFeedback] = {}
        self._extract_feedback(extractor or Extractor())
        self._validate_targets()

    @property
    def feedback_action_ids(self) -> tuple[ActionId, ...]:
        """Action IDs to exclude from the procedure sequence."""

        return tuple(self.feedback_by_action_id)

    @staticmethod
    def _bind_workers(
        worker_bindings: Iterable[tuple[object, Worker]],
    ) -> Mapping[ActionId, Worker]:
        workers_by_action_id: dict[ActionId, Worker] = {}
        for observed_actor, worker in worker_bindings:
            for action_id in getattr(observed_actor, "action_ids", ()):
                existing = workers_by_action_id.get(action_id)
                if existing is not None and existing is not worker:
                    raise ValueError(
                        f"Action {action_id!r} resolves to multiple workers"
                    )
                workers_by_action_id[action_id] = worker
        return workers_by_action_id

    def _extract_feedback(self, extractor: Extractor) -> None:
        action_summaries = [
            {"id": action.id, "action": action.action}
            for segment in self.instruction.segments
            for action in segment.actions
        ]
        for segment in self.instruction.segments:
            result = extractor.extract(
                text=segment.model_dump_json(),
                response_model=SegmentFeedback,
                system_prompt=self._feedback_prompt(segment, action_summaries),
            )
            self._add_segment_feedback(segment, result)

    def _feedback_prompt(
        self,
        segment: Segment,
        action_summaries: list[dict[str, str]],
    ) -> str:
        source_action_ids = [action.id for action in segment.actions]
        return (
            "Classify observed actions that are explicit operator feedback, not "
            "procedure work. Feedback is a reported question, issue, suggestion, "
            "or observation about a procedural action. Return an ActionFeedback "
            "only for feedback actions; omit normal work actions even if their "
            "label includes words like check, report, or observe. Set "
            "feedback_action_id to an action in this segment and step_action_id "
            "to the procedural action it concerns. Do not invent content, "
            "severity, resolutions, or action IDs. Source action IDs are "
            f"{source_action_ids!r}; available procedural-action candidates are "
            f"{action_summaries!r}. The surrounding instruction is "
            f"id={self.instruction.id!r}, name={self.instruction.name!r}."
        )

    def _add_segment_feedback(
        self,
        segment: Segment,
        result: SegmentFeedback,
    ) -> None:
        valid_source_ids = {action.id for action in segment.actions}
        for feedback in result.feedback:
            if feedback.feedback_action_id not in valid_source_ids:
                raise ValueError(
                    "Segment feedback returned an action ID outside its segment"
                )
            if feedback.feedback_action_id in self.feedback_by_action_id:
                raise ValueError("Feedback extraction returned a duplicate action ID")
            self.feedback_by_action_id[feedback.feedback_action_id] = feedback

    def _validate_targets(self) -> None:
        all_action_ids = set(self.instruction.reference_index.actions_by_id)
        feedback_action_ids = set(self.feedback_by_action_id)
        for feedback in self.feedback_by_action_id.values():
            if feedback.step_action_id not in all_action_ids:
                raise ValueError("Feedback references an unknown procedural action")
            if feedback.step_action_id in feedback_action_ids:
                raise ValueError("Feedback must reference a retained procedural action")

    def create_step_feedback(
        self,
        executions_by_action_id: Mapping[ActionId, StepExecution],
    ) -> list[StepFeedback]:
        """Bind classified feedback to its retained target execution and worker."""

        relations: list[StepFeedback] = []
        for feedback in self.feedback_by_action_id.values():
            try:
                execution = executions_by_action_id[feedback.step_action_id]
                worker = self._workers_by_action_id[feedback.feedback_action_id]
            except KeyError as error:
                raise ValueError(
                    "Feedback cannot be linked to a retained step execution or worker"
                ) from error
            relations.append(
                StepFeedback(
                    step_execution=execution,
                    reported_by=worker,
                    feedback_type=feedback.feedback_type,
                    content=feedback.content,
                    severity=feedback.severity,
                    resolved=feedback.resolved,
                    resolution=feedback.resolution,
                    source_text=self.instruction.reference_index.actions_by_id[
                        feedback.feedback_action_id
                    ].model_dump_json(),
                )
            )
        return relations
