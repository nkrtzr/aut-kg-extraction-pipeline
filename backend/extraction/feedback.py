"""Identify explicit feedback actions without delegating source IDs to an LLM."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from backend.extraction.extractor import Extractor
from backend.schemas.process_knowledge.text import ActionId, Instruction, ObservedAction


class FeedbackAssessment(BaseModel):
    """Semantic assessment of one already-known observed action."""

    is_feedback: bool = False
    feedback_type: Literal["question", "issue", "suggestion", "observation"] | None = None
    content: str | None = None
    severity: Literal["info", "warning", "critical"] | None = None
    resolved: bool = False
    resolution: str | None = None


@dataclass(frozen=True)
class FeedbackDiagnostic:
    """Feedback retained outside the graph until target linking is designed."""

    action_id: ActionId
    assessment: FeedbackAssessment


class FeedbackResolver:
    """Classify feedback actions before they become procedural steps."""

    def __init__(self, instruction: Instruction, *, extractor: Extractor | None = None) -> None:
        self.instruction = instruction
        self.assessments_by_action_id: dict[ActionId, FeedbackAssessment] = {}
        self._diagnostics: list[FeedbackDiagnostic] = []
        self._extract_feedback(extractor or Extractor())

    @property
    def feedback_action_ids(self) -> tuple[ActionId, ...]:
        """Pipeline-owned action IDs to exclude from procedure sequencing."""

        return tuple(self.assessments_by_action_id)

    @property
    def diagnostics(self) -> tuple[FeedbackDiagnostic, ...]:
        """Read-only feedback observations not represented as graph edges."""

        return tuple(self._diagnostics)

    def _extract_feedback(self, extractor: Extractor) -> None:
        for segment in self.instruction.segments:
            for action in segment.actions:
                assessment = extractor.extract(
                    text=action.model_dump_json(),
                    response_model=FeedbackAssessment,
                    system_prompt=self._feedback_prompt(action),
                )
                if assessment.is_feedback:
                    self.assessments_by_action_id[action.id] = assessment
                    self._diagnostics.append(
                        FeedbackDiagnostic(action_id=action.id, assessment=assessment)
                    )

    def _feedback_prompt(self, action: ObservedAction) -> str:
        return (
            "Classify this one observed action. Is it explicit operator feedback "
            "(a question, issue, suggestion, or observation about work) rather "
            "than procedural work? Return is_feedback=false for normal work. "
            "Do not infer feedback from safety annotations or unstated context. "
            "If is_feedback=true, provide only supported feedback attributes. "
            "Do not return action IDs, object IDs, or links to other actions. "
            f"Instruction id={self.instruction.id!r}, name={self.instruction.name!r}; "
            f"current action text={action.action!r}."
        )
