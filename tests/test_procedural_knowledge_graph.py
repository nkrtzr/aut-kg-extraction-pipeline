import pytest

from backend.schemas.process_knowledge.entities import (
    Procedure,
    ProcessParameter,
    Step,
    Tool,
)
from backend.schemas.process_knowledge.pkg import ProceduralKnowledgeGraph
from backend.schemas.process_knowledge.relations import (
    ProcedureHasStep,
    ToolRequirement,
)


def make_graph() -> tuple[ProceduralKnowledgeGraph, list[Step]]:
    graph = ProceduralKnowledgeGraph()
    steps = [Step(name=f"Step {number}") for number in range(1, 4)]
    graph.add_sequence(
        Procedure(name="Procedure"),
        {
            "action-1": steps[0],
            "action-2": steps[1],
            "action-3": steps[2],
        },
    )
    return graph, steps


def test_graph_adds_sequence_and_builds_order_relations() -> None:
    graph, steps = make_graph()

    assert graph.steps == tuple(steps)
    assert [step.step_number for step in graph.steps] == [1, 2, 3]
    assert [(order.before, order.after) for order in graph.step_orders] == [
        (steps[0], steps[1]),
        (steps[1], steps[2]),
    ]
    graph.validate()


def test_graph_assembles_completed_extraction_results() -> None:
    procedure = Procedure(name="Procedure")
    step = Step(name="Protect eyes")
    tool = Tool(name="Safety glasses")
    parameter = ProcessParameter(name="Duration", parameter_type="time")
    relation = ToolRequirement(step=step, tool=tool, parameters=[parameter])

    graph = ProceduralKnowledgeGraph.from_results(
        entities=[tool],
        procedure=procedure,
        steps_by_action_id={"action-1": step},
        process_parameters_by_action_id={"action-1": [parameter]},
        relations=[relation],
    )

    assert graph.procedure is procedure
    assert graph.steps == (step,)
    assert relation in graph.relations
    assert parameter in graph.entities


def test_remove_step_cascades_relations_parameters_and_rebuilds_order() -> None:
    graph, steps = make_graph()
    tool = Tool(name="Clamp")
    parameter = ProcessParameter(name="Force", parameter_type="force")
    graph.add_entity(tool)
    graph.add_entity(parameter)
    graph.add_relation(
        ProcedureHasStep(procedure=graph.procedure, step=steps[1])
    )
    graph.add_relation(ToolRequirement(step=steps[1], tool=tool))
    graph.process_parameters_by_action_id["action-2"] = [parameter]

    removed = graph.remove_step("action-2")

    assert removed is steps[1]
    assert graph.steps == (steps[0], steps[2])
    assert [step.step_number for step in graph.steps] == [1, 2]
    assert len(graph.step_orders) == 1
    assert graph.step_orders[0].before is steps[0]
    assert graph.step_orders[0].after is steps[2]
    assert all(entity is not removed for entity in graph.entities)
    assert all(entity is not parameter for entity in graph.entities)
    assert not any(
        isinstance(relation, ToolRequirement) for relation in graph.relations
    )
    assert not any(
        isinstance(relation, ProcedureHasStep) and relation.step is removed
        for relation in graph.relations
    )


def test_remove_step_restrict_leaves_graph_unchanged() -> None:
    graph, steps = make_graph()
    tool = Tool(name="Clamp")
    requirement = ToolRequirement(step=steps[1], tool=tool)
    graph.add_entity(tool)
    graph.add_relation(requirement)

    with pytest.raises(ValueError, match=r"1 dependent relation\(s\)"):
        graph.remove_step("action-2", dependent_policy="restrict")

    assert graph.steps == tuple(steps)
    assert requirement in graph.relations
    graph.validate()


def test_relation_must_reference_registered_entities() -> None:
    graph, steps = make_graph()

    with pytest.raises(ValueError, match="not in the graph"):
        graph.add_relation(ToolRequirement(step=steps[0], tool=Tool(name="Clamp")))
