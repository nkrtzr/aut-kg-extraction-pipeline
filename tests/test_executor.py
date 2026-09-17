from backend.extraction.executor import ProcedureExecutionAssembler
from backend.schemas.process_knowledge.entities import Procedure, Step, Worker
from backend.schemas.process_knowledge.relations import StepExecution, StepFeedback


def test_procedure_execution_aggregates_existing_execution_relations() -> None:
    procedure = Procedure(name="Assemble bracket")
    first_worker = Worker(name="Alex")
    second_worker = Worker(name="Sam")
    first_execution = StepExecution(
        step=Step(name="Assemble"),
        executed_by=first_worker,
    )
    second_execution = StepExecution(
        step=Step(name="Inspect"),
        executed_by=second_worker,
    )
    feedback = StepFeedback(
        step_execution=first_execution,
        reported_by=first_worker,
        feedback_type="issue",
        content="Bracket is scratched.",
    )

    assembler = ProcedureExecutionAssembler(
        procedure,
        [first_execution, second_execution],
        feedback=[feedback],
    )

    execution = assembler.procedure_execution
    assert execution.procedure is procedure
    assert execution.executed_by == [first_worker, second_worker]
    assert execution.step_executions == [first_execution, second_execution]
    assert execution.feedback == [feedback]
    assert execution.started_at is None
    assert execution.overall_status is None
