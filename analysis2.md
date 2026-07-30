# Current implementation state

## Overview

The repository now contains an in-memory pipeline that converts an
`Instruction` into resolved knowledge-graph entities and relations:

```text
Instruction
    ├── ActorResolver ──> ObservedActor ──> Worker
    ├── ObjectResolver ─> ObservedObject ─> Tool | Material | PPE
    └── Sequencer ──────> Procedure + Step + StepOrder
                                      │
                                      ▼
                                  Refiner
                                      │
                                      ├── ToolRequirement
                                      ├── MaterialRequirement
                                      ├── PPERequirement
                                      └── ProcessParameter
```

The deterministic parts preserve observation order and action references.
Semantic deduplication, entity enrichment, object classification, and process
parameter extraction use the existing `Extractor` and its structured LLM
responses.

## Observation schemas and instruction indexing

File: `backend/schemas/process_knowledge/text.py`

### `ObservedAction`

An action contains:

- `id`
- start and end timestamps
- an `actor_id` reference
- action name
- optional `object_id`, `instrument_id`, and `target_id` references

### `ObservedActor` and `ObservedObject`

Both observation types have their own typed `id` and contain an ordered
`action_ids` list recording every
action in which the observation occurs.

- Actors match by name.
- Objects match by name and object type.
- Their `merge()` methods extend `action_ids` while preserving order and
  removing repeated IDs.

### `Instruction`

An instruction contains `id`, `name`, and ordered segments. During
initialization it validates globally unique segment, action, actor, and object
IDs and rejects actor/object endpoints that do not resolve within the action's
segment. It then builds a read-only `reference_index` containing:

- entities by their typed IDs;
- ordered reverse actor-to-action and object-to-action references.

The compatibility `actors` and `objects` views are read-only per-segment
mappings. The helper `extract_actors()` returns actors deduplicated across the
complete instruction with their action references merged.

`VideoSegment` still numbers repeated actor and object names in observation
order.

## Resolver layer

File: `backend/extraction/resolver.py`

### Abstract `Resolver`

`Resolver[EntityT, SourceT]` contains the shared logic for:

- collecting entities from a source;
- deterministic logical deduplication;
- merging entities through their `matches()` and `merge()` methods;
- structured LLM-based semantic deduplication;
- validating that the LLM neither drops nor invents labels;
- retaining the first-observed order;
- adding instruction ID and name metadata to LLM prompts.

Resolver entity collections are exposed as tuples. Derived object/entity
bindings are invalidated whenever entities or sources change, and bindings are
also exposed as tuples so consumers cannot mutate the cached collection.

Semantic deduplication also merges the action references of entities placed in
the same semantic group.

### `ActorResolver`

`ActorResolver` collects `ObservedActor` instances from an instruction.

Important outputs and methods:

- `actors`: logically deduplicated actors;
- `semantic_deduplicate()`: LLM-assisted actor identity resolution;
- `create_workers()`: creates one `Worker` per resolved actor.

Worker extraction preserves exact actor names and order. Optional fields such
as `role` are requested only when supported by the actor name. The response is
rejected if a worker is missing, duplicated, invented, or renamed.

### `ObjectResolver`

`ObjectResolver` collects and deduplicates `ObservedObject` instances.

Important outputs and methods:

- `objects`: logically deduplicated objects;
- `semantic_deduplicate()`: LLM-assisted object identity resolution;
- `create_entities()`: classifies each object as `Tool`, `Material`, or `PPE`;
- `entity_bindings`: pairs each created entity with its reference-bearing
  `ObservedObject`.

Object classification uses both the name and object type. It requires exactly
one selected entity field and preserves the original object name. The retained
bindings allow later stages to connect entities to actions and steps.

## Sequencing

File: `backend/extraction/sequnecer.py`

`Sequencer` performs deterministic conversion from an instruction to the
procedure sequence:

- creates a `Procedure` from `Instruction.name` and `Instruction.id`;
- flattens actions in segment and action observation order;
- creates one globally numbered `Step` per action;
- retains the serialized action in `Step.source_text`;
- creates `steps_by_action_id`;
- creates a `StepOrder` between every adjacent pair of steps.

An instruction without actions produces empty step and ordering collections.

Note: the filename is currently spelled `sequnecer.py`, matching the requested
path, rather than the conventional `sequencer.py`.

## Refinement

File: `backend/extraction/refiner.py`

`Refiner` accepts a `Sequencer` and an `ObjectResolver` whose
`create_entities()` method has already been called.

Using `ObservedObject.action_ids` and `Sequencer.steps_by_action_id`, it creates:

- `ToolRequirement` for every tool/action link;
- `MaterialRequirement` for every material/action link;
- `PPERequirement` for every PPE/action link.

The combined relations are exposed through `relations`, with typed collections
available as `tool_requirements`, `material_requirements`, and
`ppe_requirements`.

The refiner then makes one structured LLM call per segment. The LLM receives
the serialized segment, valid action IDs, and instruction metadata. It may
return explicit `ProcessParameter` instances associated with action IDs.

Parameters are exposed through:

- `process_parameters`;
- `process_parameters_by_action_id`.

Parameters associated with an action using a tool are also assigned to the
matching `ToolRequirement.parameters` field. Quality requirements are
deliberately not extracted in this refinement step.

The refiner rejects:

- use before object entities have been created;
- object references to unknown actions;
- duplicate action IDs in one LLM refinement response;
- action IDs outside the segment being analyzed.

## Typical pipeline usage

```python
actor_resolver = ActorResolver(instruction)
actors = actor_resolver.semantic_deduplicate()
workers = actor_resolver.create_workers(actors)

object_resolver = ObjectResolver(instruction)
objects = object_resolver.semantic_deduplicate()
physical_entities = object_resolver.create_entities(objects)

sequencer = Sequencer(instruction)
refiner = Refiner(sequencer, object_resolver)
```

The LLM calls require the configured Ollama service unless an `Extractor`
replacement is injected for testing.

## Verification

Focused verification currently passes:

- 19 tests across resolver, sequencer, refiner, and text-schema behavior;
- Ruff checks for the changed implementation and test files.

The tests use injected fake extractors, so they verify prompts, structured
response handling, validation, reference merging, ordering, classification,
and relation construction without requiring a running LLM.

## Current limitations and follow-up work

- The pipeline is in-memory and has no graph persistence layer.
- No top-level orchestrator currently runs all stages as one operation.
- The LLM-dependent production path is not covered by an Ollama integration
  test.
- Semantic matching relies on labels and instruction metadata rather than the
  complete segment context.
- Process parameters can currently be attached directly only to
  `ToolRequirement`, because the material and PPE requirement schemas have no
  parameter field.
- Several current implementation and test files are untracked in Git and need
  to be added before committing.
