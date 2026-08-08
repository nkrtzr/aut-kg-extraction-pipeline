"""Build procedure and step sequencing entities from an instruction."""

from backend.schemas.process_knowledge.entities import Procedure, Step
from backend.schemas.process_knowledge.relations import ProcedureHasStep, StepOrder
from backend.schemas.process_knowledge.text import Instruction, ObservedAction


class Sequencer:
    """Convert an instruction's ordered actions into a procedure sequence."""

    def __init__(self, instruction: Instruction):
        self.instruction = instruction
        self.procedure = self._create_procedure(instruction)
        self.actions = [
            action
            for segment in instruction.segments
            for action in segment.actions
        ]
        self.steps = self._create_steps(self.actions)
        self.steps_by_action_id = dict(
            zip(
                (action.id for action in self.actions),
                self.steps,
                strict=True,
            )
        )
        self.procedure_has_steps = self._create_procedure_has_steps(
            self.procedure,
            self.steps,
        )
        self.step_orders = self._create_step_orders(self.steps)

    @staticmethod
    def _create_procedure(instruction: Instruction) -> Procedure:
        """Create the procedure represented by the instruction."""

        return Procedure(
            name=instruction.name,
            procedure_id=instruction.id,
        )

    @staticmethod
    def _create_steps(actions: list[ObservedAction]) -> list[Step]:
        """Create one globally numbered step for every observed action."""

        return [
            Step(
                name=action.action,
                step_number=index,
                source_text=action.model_dump_json(),
            )
            for index, action in enumerate(actions, start=1)
        ]

    @staticmethod
    def _create_step_orders(steps: list[Step]) -> list[StepOrder]:
        """Connect every adjacent pair of steps in observation order."""

        return [
            StepOrder(before=before, after=after)
            for before, after in zip(steps, steps[1:])
        ]

    @staticmethod
    def _create_procedure_has_steps(
        procedure: Procedure,
        steps: list[Step],
    ) -> list[ProcedureHasStep]:
        """Connect the procedure to every step in procedural order."""

        return [
            ProcedureHasStep(procedure=procedure, step=step)
            for step in steps
        ]
