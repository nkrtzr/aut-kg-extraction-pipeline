"""Schemas for observations extracted from text and video segments."""

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, NewType, Sequence

from pydantic import BaseModel, Field, model_validator

ActionId = NewType("ActionId", str)
ActorId = NewType("ActorId", str)
ObjectId = NewType("ObjectId", str)
SegmentId = NewType("SegmentId", str)


class ObservedAction(BaseModel):
    """An action extracted from a text or video source.

    Actor and entity references use typed observation IDs. Display names are
    never used as foreign keys. Timestamps are optional for text sources.
    """

    id: ActionId
    start_time_ms: int | None = None
    end_time_ms: int | None = None

    actor_id: ActorId
    action: str
    object_id: ObjectId | None = None
    instrument_id: ObjectId | None = None
    target_id: ObjectId | None = None


class ObservedObject(BaseModel):
    """An object observed in a segment."""

    id: ObjectId
    name: str
    object_type: str
    action_ids: list[ActionId] = Field(default_factory=list)

    def matches(self, other: object) -> bool:
        """Match objects by their observed name and type."""

        return (
            isinstance(other, ObservedObject)
            and self.name == other.name
            and self.object_type == other.object_type
        )

    def merge(self, other: "ObservedObject") -> None:
        """Merge action references while preserving observation order."""

        self.action_ids = list(dict.fromkeys([*self.action_ids, *other.action_ids]))


class ObservedActor(BaseModel):
    """An actor observed in a video segment."""

    id: ActorId
    name: str
    action_ids: list[ActionId] = Field(default_factory=list)

    def matches(self, other: object) -> bool:
        """Match actors by their observed name."""

        return isinstance(other, ObservedActor) and self.name == other.name

    def merge(self, other: "ObservedActor") -> None:
        """Merge action references while preserving observation order."""

        self.action_ids = list(dict.fromkeys([*self.action_ids, *other.action_ids]))


class Segment(BaseModel):
    """A text-derived segment containing observations to be processed."""

    id: SegmentId
    scene: str
    actions: list[ObservedAction]
    actors: list[ObservedActor]
    objects: list[ObservedObject]
    uncertainties: list[str]


class VideoSegment(Segment):
    """A time-bounded segment produced by the video-processing pipeline.

    Repeated object and actor names are numbered after validation so that every
    observed entity has a unique name within the segment.
    """

    start_time_ms: int
    end_time_ms: int

    scene: str
    actions: list[ObservedAction]
    actors: list[ObservedActor]
    objects: list[ObservedObject]
    uncertainties: list[str]

    @model_validator(mode="after")
    def require_action_timestamps(self) -> "VideoSegment":
        """Require timestamps for every action in a video segment."""

        for action in self.actions:
            if (
                action.start_time_ms is None
                or action.end_time_ms is None
            ):
                raise ValueError(
                    "Video actions must contain start and end timestamps"
                )

        return self

    @model_validator(mode="after")
    def number_repeated_entities(self) -> "VideoSegment":
        """Number repeated object and actor names in their observation order."""

        self._number_repeated_names(self.objects)
        self._number_repeated_names(self.actors)
        return self

    @staticmethod
    def _number_repeated_names(
        entities: list[ObservedObject] | list[ObservedActor],
    ) -> None:
        """Append one-based indices to every occurrence of a repeated name."""

        name_counts = Counter(entity.name for entity in entities)
        next_number: Counter[str] = Counter()

        for entity in entities:
            original_name = entity.name
            if name_counts[original_name] > 1:
                next_number[original_name] += 1
                entity.name = f"{original_name}_{next_number[original_name]}"


@dataclass(frozen=True)
class InstructionReferenceIndex:
    """Read-only instruction-wide identity and reverse-reference indexes."""

    segments_by_id: Mapping[SegmentId, Segment]
    actions_by_id: Mapping[ActionId, ObservedAction]
    actors_by_id: Mapping[ActorId, ObservedActor]
    objects_by_id: Mapping[ObjectId, ObservedObject]
    actor_action_ids: Mapping[ActorId, tuple[ActionId, ...]]
    object_action_ids: Mapping[ObjectId, tuple[ActionId, ...]]


class Instruction:
    """A collection of segments with observations indexed by segment ID.

    The instruction owns a validated, read-only reference index. Every action
    endpoint must resolve to an actor or object declared in the same segment.
    """

    def __init__(
        self,
        id: str,
        name: str,
        segments: list[Segment],
    ):
        """Initialize an instruction and build its per-segment lookup maps."""

        self.id = id
        self.name = name
        self.segments = segments
        self.reference_index = self._build_reference_index(segments)
        self.objects = MappingProxyType(
            {segment.id: tuple(segment.objects) for segment in segments}
        )
        self.actors = MappingProxyType(
            {segment.id: tuple(segment.actors) for segment in segments}
        )

    @classmethod
    def from_events(
        cls,
        events: Sequence[Mapping[str, Any]],
        *,
        actor_name: str = "operator",
    ) -> tuple["Instruction", list[str]]:
        """Build an instruction from the standard segmented video-event format.

        The source format has no actor annotations, so each event receives one
        placeholder actor. Object endpoint roles are not annotated either;
        their source order maps to object, instrument, then target.
        """

        if not events:
            raise ValueError("Expected a non-empty top-level JSON list.")

        segments: list[VideoSegment] = []
        conversion_warnings: list[str] = []

        def seconds_to_ms(value: Any, *, fallback: int) -> int:
            return fallback if value is None else round(float(value) * 1000)

        for event in events:
            event_id = str(event["event_id"])
            start_ms = seconds_to_ms(event.get("start_sec"), fallback=0)
            end_ms = seconds_to_ms(event.get("end_sec"), fallback=start_ms)
            actor_id = ActorId(f"{event_id}:actor:1")
            actor = ObservedActor(id=actor_id, name=actor_name)
            observed_objects: list[ObservedObject] = []
            object_id_by_key: dict[tuple[str, str], ObjectId] = {}
            actions: list[ObservedAction] = []

            for action_index, raw_action in enumerate(event.get("actions", []), start=1):
                action_id = ActionId(f"{event_id}:action:{action_index}")
                action_object_ids: list[ObjectId] = []
                for raw_object in raw_action.get("objects", []):
                    name = str(raw_object["name"])
                    object_type = str(raw_object["type"])
                    key = (name, object_type)
                    object_id = object_id_by_key.get(key)
                    if object_id is None:
                        object_id = ObjectId(
                            f"{event_id}:object:{len(observed_objects) + 1}"
                        )
                        object_id_by_key[key] = object_id
                        observed_objects.append(
                            ObservedObject(
                                id=object_id,
                                name=name,
                                object_type=object_type,
                            )
                        )
                    action_object_ids.append(object_id)

                action_object_ids = list(dict.fromkeys(action_object_ids))
                if len(action_object_ids) > 3:
                    names_by_id = {obj.id: obj.name for obj in observed_objects}
                    conversion_warnings.append(
                        f"{action_id}: current ObservedAction schema can reference "
                        "only 3 objects; not linked to the action: "
                        f"{[names_by_id[obj_id] for obj_id in action_object_ids[3:]]}"
                    )

                action_start_ms = seconds_to_ms(
                    raw_action.get("start_sec"), fallback=start_ms
                )
                action_end_ms = seconds_to_ms(
                    raw_action.get("end_sec"), fallback=end_ms
                )
                actions.append(
                    ObservedAction(
                        id=action_id,
                        start_time_ms=action_start_ms,
                        end_time_ms=action_end_ms,
                        actor_id=actor_id,
                        action=str(raw_action["description"]),
                        object_id=(action_object_ids[0] if action_object_ids else None),
                        instrument_id=(
                            action_object_ids[1]
                            if len(action_object_ids) >= 2
                            else None
                        ),
                        target_id=(
                            action_object_ids[2]
                            if len(action_object_ids) >= 3
                            else None
                        ),
                    )
                )

            uncertainty = event.get("uncertainty") or {}
            uncertainty_note = uncertainty.get("note", "")
            uncertainties = [str(uncertainty_note)] if uncertainty_note else []
            for warning in event.get("warnings", []):
                warning_type = warning.get("type", "warning")
                description = warning.get("description", "")
                uncertainties.append(f"{warning_type}: {description}".strip())

            segments.append(
                VideoSegment(
                    id=SegmentId(event_id),
                    start_time_ms=start_ms,
                    end_time_ms=end_ms,
                    scene=str(event.get("step_description", event_id)),
                    actions=actions,
                    actors=[actor],
                    objects=observed_objects,
                    uncertainties=uncertainties,
                )
            )

        video_ids = {
            str(event["video_id"])
            for event in events
            if event.get("video_id") is not None
        }
        if len(video_ids) != 1:
            raise ValueError(
                f"Expected exactly one video_id for one Instruction, got {video_ids}"
            )
        instruction_id = next(iter(video_ids))
        return (
            cls(id=instruction_id, name=instruction_id, segments=segments),
            conversion_warnings,
        )

    @staticmethod
    def _duplicates(values: list[str]) -> list[str]:
        return sorted(value for value, count in Counter(values).items() if count > 1)

    @classmethod
    def _build_reference_index(
        cls,
        segments: list[Segment],
    ) -> InstructionReferenceIndex:
        """Validate identities and endpoints, then build all reference maps."""

        duplicate_ids = {
            "segment": cls._duplicates([segment.id for segment in segments]),
            "action": cls._duplicates(
                [action.id for segment in segments for action in segment.actions]
            ),
            "actor": cls._duplicates(
                [actor.id for segment in segments for actor in segment.actors]
            ),
            "object": cls._duplicates(
                [
                    observed_object.id
                    for segment in segments
                    for observed_object in segment.objects
                ]
            ),
        }
        for kind, duplicates in duplicate_ids.items():
            if duplicates:
                scope = "globally " if kind != "segment" else ""
                raise ValueError(
                    f"Instruction {kind} IDs must be {scope}unique; "
                    f"duplicates: {duplicates!r}"
                )

        segments_by_id = {segment.id: segment for segment in segments}
        actions_by_id: dict[ActionId, ObservedAction] = {}
        actors_by_id: dict[ActorId, ObservedActor] = {}
        objects_by_id: dict[ObjectId, ObservedObject] = {}
        actor_action_ids: dict[ActorId, list[ActionId]] = {}
        object_action_ids: dict[ObjectId, list[ActionId]] = {}

        for segment in segments:
            segment_actor_ids = {actor.id for actor in segment.actors}
            segment_object_ids = {
                observed_object.id for observed_object in segment.objects
            }
            actors_by_id.update((actor.id, actor) for actor in segment.actors)
            objects_by_id.update(
                (observed_object.id, observed_object)
                for observed_object in segment.objects
            )
            actor_action_ids.update((actor.id, []) for actor in segment.actors)
            object_action_ids.update(
                (observed_object.id, []) for observed_object in segment.objects
            )

            for action in segment.actions:
                if action.actor_id not in segment_actor_ids:
                    raise ValueError(
                        f"Action {action.id!r} references unknown actor "
                        f"{action.actor_id!r} in segment {segment.id!r}"
                    )
                object_refs = (
                    action.object_id,
                    action.instrument_id,
                    action.target_id,
                )
                unknown_object_ids = sorted(
                    {
                        object_id
                        for object_id in object_refs
                        if object_id is not None and object_id not in segment_object_ids
                    }
                )
                if unknown_object_ids:
                    raise ValueError(
                        f"Action {action.id!r} references unknown objects "
                        f"{unknown_object_ids!r} in segment {segment.id!r}"
                    )

                actions_by_id[action.id] = action
                actor_action_ids[action.actor_id].append(action.id)
                for object_id in dict.fromkeys(object_refs):
                    if object_id is not None:
                        object_action_ids[object_id].append(action.id)

        for actor_id, actor in actors_by_id.items():
            actor.action_ids = list(actor_action_ids.get(actor_id, ()))
        for object_id, observed_object in objects_by_id.items():
            observed_object.action_ids = list(object_action_ids.get(object_id, ()))

        return InstructionReferenceIndex(
            segments_by_id=MappingProxyType(segments_by_id),
            actions_by_id=MappingProxyType(actions_by_id),
            actors_by_id=MappingProxyType(actors_by_id),
            objects_by_id=MappingProxyType(objects_by_id),
            actor_action_ids=MappingProxyType(
                {key: tuple(value) for key, value in actor_action_ids.items()}
            ),
            object_action_ids=MappingProxyType(
                {key: tuple(value) for key, value in object_action_ids.items()}
            ),
        )

    def extract_actors(self) -> list[ObservedActor]:
        """Extract a de-duplicated list of actors across all segments."""

        actors_by_name: dict[str, ObservedActor] = {}
        for segment_actors in self.actors.values():
            for actor in segment_actors:
                existing = actors_by_name.get(actor.name)
                if existing is None:
                    actors_by_name[actor.name] = actor.model_copy(deep=True)
                else:
                    existing.merge(actor)
        return list(actors_by_name.values())
