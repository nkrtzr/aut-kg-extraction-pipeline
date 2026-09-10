"""Convert text and video descriptions into validated segments."""

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from backend.extraction.extractor import Extractor
from backend.schemas.process_knowledge.text import (
    Instruction,
    ObservedAction,
    ObservedActor,
    ObservedObject,
    Segment,
    VideoSegment,
)

SourceType = Literal["video", "text"]
TextFormat = Literal["structured", "free_form"]


class ExtractedAction(BaseModel):
    """One action extracted from a text or video source."""

    actor: str = "unspecified actor"
    action: str
    object_name: str | None = None
    instrument_name: str | None = None
    target_name: str | None = None
    start_time: str | None = None
    end_time: str | None = None

    @field_validator("actor", mode="before")
    @classmethod
    def replace_missing_actor(cls, value: object) -> object:
        """Replace a missing actor with a transparent placeholder."""

        if value is None:
            return "unspecified actor"

        if isinstance(value, str) and not value.strip():
            return "unspecified actor"

        return value

    @field_validator(
        "object_name",
        "instrument_name",
        "target_name",
        "start_time",
        "end_time",
        mode="before",
    )
    @classmethod
    def replace_null_strings(cls, value: object) -> object:
        """Convert textual null values into real null values."""

        if not isinstance(value, str):
            return value

        cleaned = value.strip()

        if cleaned.casefold() in {
            "",
            "null",
            "none",
            "n/a",
            "not applicable",
        }:
            return None

        return cleaned


class ExtractedObject(BaseModel):
    """One object extracted from an input segment."""

    name: str
    object_type: str


class SegmentDraft(BaseModel):
    """Intermediate information used to build a segment."""

    segment_id: str
    scene: str
    actions: list[ExtractedAction] = Field(default_factory=list)
    objects: list[ExtractedObject] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    start_time: str | None = None
    end_time: str | None = None


class TextInstructionDraft(BaseModel):
    """Normalized representation of a free-form instruction."""

    instruction_id: str = "text-instruction"
    instruction_name: str = "Text instruction"
    segments: list[SegmentDraft] = Field(default_factory=list)


class Segmenter:
    """Convert video-derived or text-derived input into an Instruction."""

    SECTION_NAMES = (
        "Video",
        "Instruction",
        "Instruction ID",
        "Instruction name",
        "Segment",
        "Time",
        "Scene",
        "Actions",
        "Objects",
        "Uncertainty",
    )

    UNCERTAINTY_MARKERS = (
        "not stated",
        "not specified",
        "not provided",
        "not known",
        "is unknown",
        "are unknown",
        "is unclear",
        "are unclear",
        "cannot be determined",
        "cannot determine",
        "uncertain",
    )

    def __init__(self, *, extractor: Extractor | None = None):
        self.extractor = extractor or Extractor()

    def segment(
        self,
        text: str,
        *,
        source_type: SourceType = "video",
        text_format: TextFormat = "structured",
        instruction_id: str | None = None,
        instruction_name: str | None = None,
    ) -> Instruction:
        """Convert the selected input type into an Instruction."""

        if not text.strip():
            raise ValueError("Input text must not be empty")

        if source_type == "video":
            if text_format != "structured":
                raise ValueError(
                    "Video input must use the structured format"
                )

            return self._segment_video(text)

        if source_type == "text":
            if text_format == "structured":
                return self._segment_structured_text(
                    text,
                    instruction_id=instruction_id,
                    instruction_name=instruction_name,
                )

            if text_format == "free_form":
                return self._segment_free_form_text(
                    text,
                    instruction_id=instruction_id,
                    instruction_name=instruction_name,
                )

            raise ValueError(
                "text_format must be 'structured' or 'free_form'"
            )

        raise ValueError(
            "source_type must be 'video' or 'text'"
        )

    def _segment_video(self, text: str) -> Instruction:
        """Parse one structured video segment."""

        video_id = self._read_header(text, "Video")
        segment_id = self._read_header(text, "Segment")

        instruction_name = self._read_header(
            text,
            "Instruction name",
            required=False,
        )

        if instruction_name is None:
            instruction_name = video_id

        time_range = self._read_header(text, "Time")
        start_time, end_time = self._split_time_range(time_range)

        draft = SegmentDraft(
            segment_id=segment_id,
            start_time=start_time,
            end_time=end_time,
            scene=self._read_section(text, "Scene"),
            actions=self._extract_actions(
                self._read_section(text, "Actions"),
                require_timestamps=True,
            ),
            objects=self._parse_objects(
                self._read_section(text, "Objects")
            ),
            uncertainties=self._parse_bullets(
                self._read_section(
                    text,
                    "Uncertainty",
                    required=False,
                )
                or ""
            ),
        )

        return Instruction(
            id=video_id,
            name=instruction_name,
            segments=[self._build_video_segment(draft)],
        )

    def _segment_structured_text(
        self,
        text: str,
        *,
        instruction_id: str | None,
        instruction_name: str | None,
    ) -> Instruction:
        """Parse one structured text segment without timestamps."""

        parsed_instruction_id = (
            instruction_id
            or self._read_header(
                text,
                "Instruction ID",
                required=False,
            )
            or self._read_header(
                text,
                "Instruction",
                required=False,
            )
            or "text-instruction"
        )

        parsed_instruction_name = (
            instruction_name
            or self._read_header(
                text,
                "Instruction name",
                required=False,
            )
            or parsed_instruction_id
        )

        segment_id = (
            self._read_header(
                text,
                "Segment",
                required=False,
            )
            or "text-segment-1"
        )

        draft = SegmentDraft(
            segment_id=segment_id,
            scene=self._read_section(text, "Scene"),
            actions=self._extract_actions(
                self._read_section(text, "Actions"),
                require_timestamps=False,
            ),
            objects=self._parse_objects(
                self._read_section(text, "Objects")
            ),
            uncertainties=self._parse_bullets(
                self._read_section(
                    text,
                    "Uncertainty",
                    required=False,
                )
                or ""
            ),
        )

        return Instruction(
            id=parsed_instruction_id,
            name=parsed_instruction_name,
            segments=[self._build_text_segment(draft)],
        )

    def _segment_free_form_text(
        self,
        text: str,
        *,
        instruction_id: str | None,
        instruction_name: str | None,
    ) -> Instruction:
        """Normalize free-form text into text-derived segments."""

        normalized = self.extractor.extract(
            text=text,
            response_model=TextInstructionDraft,
            system_prompt=self._normalization_prompt(),
        )

        if not self._normalized_output_is_usable(normalized):
            try:
                normalized = self.extractor.extract(
                    text=text,
                    response_model=TextInstructionDraft,
                    system_prompt=self._normalization_retry_prompt(),
                )
            except StopIteration:
                pass

        if not self._normalized_output_is_usable(normalized):
            return self._segment_free_form_by_sentences(
                text,
                instruction_id=instruction_id,
                instruction_name=instruction_name,
            )

        final_instruction_id = (
            instruction_id
            or normalized.instruction_id
            or "text-instruction"
        )

        final_instruction_name = (
            instruction_name
            or normalized.instruction_name
            or final_instruction_id
        )

        segments = [
            self._build_text_segment(
                self._prepare_text_draft(
                    draft,
                    index=index,
                )
            )
            for index, draft in enumerate(
                normalized.segments,
                start=1,
            )
        ]

        return Instruction(
            id=final_instruction_id,
            name=final_instruction_name,
            segments=segments,
        )

    @classmethod
    def _normalized_output_is_usable(
        cls,
        normalized: TextInstructionDraft,
    ) -> bool:
        """Reject structurally valid but unusable LLM output."""

        if not normalized.segments:
            return False

        segment_ids = [
            segment.segment_id.strip()
            for segment in normalized.segments
        ]

        if any(not segment_id for segment_id in segment_ids):
            return False

        if len(set(segment_ids)) != len(segment_ids):
            return False

        actions = [
            action
            for segment in normalized.segments
            for action in segment.actions
        ]

        if not actions:
            return False

        if any(
            not segment.actions
            for segment in normalized.segments
        ):
            return False

        has_entity_roles = any(
            action.object_name is not None
            or action.instrument_name is not None
            or action.target_name is not None
            for action in actions
        )

        if not has_entity_roles:
            return False

        for segment in normalized.segments:
            evidence = " ".join(
                [
                    segment.scene,
                    *(
                        action.action
                        for action in segment.actions
                    ),
                ]
            )

            if (
                cls._is_uncertainty_sentence(evidence)
                and not segment.uncertainties
            ):
                return False

        return True

    def _segment_free_form_by_sentences(
        self,
        text: str,
        *,
        instruction_id: str | None,
        instruction_name: str | None,
    ) -> Instruction:
        """Build one segment through simple sentence-level extraction."""

        sentences = self._split_free_form_sentences(text)

        uncertainties = [
            sentence
            for sentence in sentences
            if self._is_uncertainty_sentence(sentence)
        ]

        process_sentences = [
            sentence
            for sentence in sentences
            if not self._is_uncertainty_sentence(sentence)
        ]

        action_blocks = [
            clause
            for sentence in process_sentences
            for clause in self._split_compound_action_sentence(sentence)
            if clause
        ]

        if not action_blocks:
            raise ValueError(
                "Free-form text did not contain any process action"
            )

        actions: list[ExtractedAction] = []

        try:
            for block in action_blocks:
                extracted = self.extractor.extract(
                    text=block,
                    response_model=ExtractedAction,
                    system_prompt=self._action_prompt(
                        require_timestamps=False,
                    ),
                )

                corrected = self._correct_action_roles(
                    extracted,
                    block,
                )

                actions.append(
                    corrected.model_copy(
                        update={
                            "start_time": None,
                            "end_time": None,
                        }
                    )
                )
        except StopIteration as error:
            raise ValueError(
                "Free-form text did not produce any segments "
                "after all extraction attempts"
            ) from error

        if not actions:
            raise ValueError(
                "Free-form text did not produce any actions"
            )

        draft = SegmentDraft(
            segment_id="text-segment-1",
            scene=" ".join(process_sentences),
            actions=actions,
            objects=[],
            uncertainties=uncertainties,
        )

        prepared = self._prepare_text_draft(
            draft,
            index=1,
        )

        final_instruction_id = (
            instruction_id or "text-instruction"
        )

        final_instruction_name = (
            instruction_name or final_instruction_id
        )

        return Instruction(
            id=final_instruction_id,
            name=final_instruction_name,
            segments=[self._build_text_segment(prepared)],
        )

    @classmethod
    def _prepare_text_draft(
        cls,
        draft: SegmentDraft,
        *,
        index: int,
    ) -> SegmentDraft:
        """Prepare a text segment for final graph construction."""

        segment_id = draft.segment_id.strip()

        if not segment_id:
            segment_id = f"text-segment-{index}"

        resolved_actions = cls._resolve_text_actors(
            draft.actions
        )

        actions = [
            action.model_copy(
                update={
                    "start_time": None,
                    "end_time": None,
                }
            )
            for action in resolved_actions
        ]

        objects = cls._complete_action_objects(
            draft.objects,
            actions,
        )

        return draft.model_copy(
            update={
                "segment_id": segment_id,
                "start_time": None,
                "end_time": None,
                "actions": actions,
                "objects": objects,
            }
        )

    @classmethod
    def _resolve_text_actors(
        cls,
        actions: list[ExtractedAction],
    ) -> list[ExtractedAction]:
        """Reuse one known actor when another action omits it."""

        placeholder_key = cls._name_key(
            "unspecified actor"
        )

        known_actors = list(
            dict.fromkeys(
                action.actor.strip()
                for action in actions
                if cls._name_key(action.actor)
                != placeholder_key
            )
        )

        if len(known_actors) != 1:
            return actions

        known_actor = known_actors[0]

        return [
            action.model_copy(
                update={"actor": known_actor}
            )
            if cls._name_key(action.actor)
            == placeholder_key
            else action
            for action in actions
        ]

    @classmethod
    def _complete_action_objects(
        cls,
        objects: list[ExtractedObject],
        actions: list[ExtractedAction],
    ) -> list[ExtractedObject]:
        """Add referenced entities omitted from the object list."""

        completed = list(objects)

        known_names = {
            cls._name_key(item.name)
            for item in completed
        }

        for action in actions:
            references = (
                (action.object_name, "object"),
                (action.instrument_name, "tool"),
                (action.target_name, "target"),
            )

            for name, object_type in references:
                if name is None:
                    continue

                normalized_name = name.strip()

                if not normalized_name:
                    continue

                key = cls._name_key(normalized_name)

                if key in known_names:
                    continue

                completed.append(
                    ExtractedObject(
                        name=normalized_name,
                        object_type=object_type,
                    )
                )

                known_names.add(key)

        return completed

    @staticmethod
    def _split_free_form_sentences(
        text: str,
    ) -> list[str]:
        """Split free-form input into non-empty sentences."""

        normalized = re.sub(
            r"\s+",
            " ",
            text.strip(),
        )

        if not normalized:
            return []

        parts = re.split(
            r"(?<=[.!?])\s+",
            normalized,
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    @classmethod
    def _is_uncertainty_sentence(
        cls,
        sentence: str,
    ) -> bool:
        """Identify explicitly uncertain or missing information."""

        normalized = sentence.casefold()

        return any(
            marker in normalized
            for marker in cls.UNCERTAINTY_MARKERS
        )

    @staticmethod
    def _split_compound_action_sentence(
        sentence: str,
    ) -> list[str]:
        """Split simple coordinated action clauses."""

        cleaned = re.sub(
            r"^(?:after that|then|next|finally),?\s+",
            "",
            sentence.strip(),
            flags=re.IGNORECASE,
        )

        parts = re.split(
            (
                r"\s+and\s+"
                r"(?=(?:then\s+)?"
                r"[A-Za-z]+(?:s|es|ed|ing)\s+"
                r"(?:the|a|an)\b)"
            ),
            cleaned,
            flags=re.IGNORECASE,
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    @staticmethod
    def _read_header(
        text: str,
        label: str,
        *,
        required: bool = True,
    ) -> str | None:
        pattern = (
            rf"(?im)^\s*{re.escape(label)}"
            rf"\s*:\s*(.+?)\s*$"
        )

        match = re.search(pattern, text)

        if match is not None:
            return match.group(1).strip()

        if required:
            raise ValueError(
                f"Missing required header: {label}"
            )

        return None

    @classmethod
    def _read_section(
        cls,
        text: str,
        label: str,
        *,
        required: bool = True,
    ) -> str | None:
        section_names = "|".join(
            re.escape(name)
            for name in cls.SECTION_NAMES
        )

        pattern = (
            rf"(?ims)^\s*{re.escape(label)}\s*:\s*"
            rf"(.*?)(?=^\s*(?:{section_names})\s*:|\Z)"
        )

        match = re.search(pattern, text)

        if match is not None:
            value = match.group(1).strip()

            if value:
                return value

        if required:
            raise ValueError(
                f"Missing required section: {label}"
            )

        return None

    @staticmethod
    def _split_time_range(
        value: str,
    ) -> tuple[str, str]:
        parts = re.split(
            r"\s+-\s+",
            value.strip(),
            maxsplit=1,
        )

        if len(parts) != 2:
            raise ValueError(
                f"Invalid time range: {value!r}"
            )

        return parts[0], parts[1]

    @staticmethod
    def _split_actions(value: str) -> list[str]:
        blocks = re.findall(
            r"(?ms)^\s*\d+\.\s*(.*?)(?=^\s*\d+\.\s*|\Z)",
            value,
        )

        if blocks:
            return [
                block.strip()
                for block in blocks
                if block.strip()
            ]

        if value.strip():
            return [value.strip()]

        return []

    def _extract_actions(
        self,
        value: str,
        *,
        require_timestamps: bool,
    ) -> list[ExtractedAction]:
        actions: list[ExtractedAction] = []

        for block in self._split_actions(value):
            extracted = self.extractor.extract(
                text=block,
                response_model=ExtractedAction,
                system_prompt=self._action_prompt(
                    require_timestamps=require_timestamps,
                ),
            )

            corrected = self._correct_action_roles(
                extracted,
                block,
            )

            if require_timestamps:
                self._validate_extracted_timestamps(
                    corrected
                )
            else:
                corrected = corrected.model_copy(
                    update={
                        "start_time": None,
                        "end_time": None,
                    }
                )

            actions.append(corrected)

        return actions

    @staticmethod
    def _validate_extracted_timestamps(
        action: ExtractedAction,
    ) -> None:
        if (
            action.start_time is None
            or action.end_time is None
        ):
            raise ValueError(
                "Video actions must contain start and end timestamps"
            )

    @classmethod
    def _correct_action_roles(
        cls,
        action: ExtractedAction,
        source_text: str,
    ) -> ExtractedAction:
        """Correct common role mistakes from a small local model."""

        action_name = action.action.casefold()

        inspection_verbs = (
            "inspect",
            "check",
            "examine",
            "observe",
        )

        is_inspection = any(
            verb in action_name
            for verb in inspection_verbs
        )

        if is_inspection:
            if (
                action.object_name is None
                and action.target_name is not None
            ):
                return action.model_copy(
                    update={
                        "object_name": action.target_name,
                        "target_name": None,
                    }
                )

            if action.object_name is not None:
                return action.model_copy(
                    update={"target_name": None}
                )

        target_verbs = (
            "tighten",
            "place",
            "attach",
            "mount",
            "install",
            "insert",
            "move",
        )

        needs_target = any(
            verb in action_name
            for verb in target_verbs
        )

        if needs_target and action.target_name is None:
            target_name = cls._target_from_text(
                source_text
            )

            if target_name is not None:
                return action.model_copy(
                    update={"target_name": target_name}
                )

        return action

    @staticmethod
    def _target_from_text(
        source_text: str,
    ) -> str | None:
        pattern = (
            r"\b(?:on|onto|into|to)\s+"
            r"(?:the\s+)?"
            r"(?P<target>[^.\n]+?)"
            r"(?=\s+using\b|[.\n]|$)"
        )

        match = re.search(
            pattern,
            source_text,
            flags=re.IGNORECASE,
        )

        if match is None:
            return None

        return match.group("target").strip()

    @staticmethod
    def _parse_objects(
        value: str,
    ) -> list[ExtractedObject]:
        objects: list[ExtractedObject] = []

        for line in value.splitlines():
            item = line.strip().lstrip("-*").strip()

            if not item:
                continue

            if ":" not in item:
                raise ValueError(
                    f"Invalid object entry: {line!r}"
                )

            name, object_type = item.split(
                ":",
                maxsplit=1,
            )

            objects.append(
                ExtractedObject(
                    name=name.strip(),
                    object_type=object_type.strip(),
                )
            )

        return objects

    @staticmethod
    def _parse_bullets(value: str) -> list[str]:
        return [
            line.strip().lstrip("-*").strip()
            for line in value.splitlines()
            if line.strip().lstrip("-*").strip()
        ]

    @staticmethod
    def _timestamp_to_ms(value: str) -> int:
        pattern = (
            r"^(?:(?P<hours>\d{1,2}):)?"
            r"(?P<minutes>\d{1,2}):"
            r"(?P<seconds>\d{1,2})"
            r"(?:[.,](?P<milliseconds>\d{1,3}))?$"
        )

        match = re.fullmatch(
            pattern,
            value.strip(),
        )

        if match is None:
            raise ValueError(
                f"Invalid timestamp: {value!r}"
            )

        hours = int(match.group("hours") or 0)
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))

        milliseconds = int(
            (match.group("milliseconds") or "0").ljust(
                3,
                "0",
            )
        )

        if minutes >= 60 or seconds >= 60:
            raise ValueError(
                f"Invalid timestamp: {value!r}"
            )

        return (
            hours * 3_600_000
            + minutes * 60_000
            + seconds * 1_000
            + milliseconds
        )

    @staticmethod
    def _name_key(value: str) -> str:
        normalized = " ".join(
            value.casefold().split()
        )

        return re.sub(
            r"^(the|a|an)\s+",
            "",
            normalized,
        )

    def _build_text_segment(
        self,
        draft: SegmentDraft,
    ) -> Segment:
        actors, actor_ids = self._build_actors(
            draft.segment_id,
            draft.actions,
        )

        objects, object_ids = self._build_objects(
            draft.segment_id,
            draft.objects,
        )

        actions = self._build_actions(
            segment_id=draft.segment_id,
            extracted_actions=draft.actions,
            actor_ids=actor_ids,
            object_ids=object_ids,
            include_timestamps=False,
        )

        return Segment(
            id=draft.segment_id,
            scene=draft.scene,
            actions=actions,
            actors=actors,
            objects=objects,
            uncertainties=draft.uncertainties,
        )

    def _build_video_segment(
        self,
        draft: SegmentDraft,
    ) -> VideoSegment:
        if (
            draft.start_time is None
            or draft.end_time is None
        ):
            raise ValueError(
                "Video segment must contain start and end timestamps"
            )

        actors, actor_ids = self._build_actors(
            draft.segment_id,
            draft.actions,
        )

        objects, object_ids = self._build_objects(
            draft.segment_id,
            draft.objects,
        )

        actions = self._build_actions(
            segment_id=draft.segment_id,
            extracted_actions=draft.actions,
            actor_ids=actor_ids,
            object_ids=object_ids,
            include_timestamps=True,
        )

        return VideoSegment(
            id=draft.segment_id,
            start_time_ms=self._timestamp_to_ms(
                draft.start_time
            ),
            end_time_ms=self._timestamp_to_ms(
                draft.end_time
            ),
            scene=draft.scene,
            actions=actions,
            actors=actors,
            objects=objects,
            uncertainties=draft.uncertainties,
        )

    def _build_actors(
        self,
        segment_id: str,
        actions: list[ExtractedAction],
    ) -> tuple[list[ObservedActor], dict[str, str]]:
        actor_names = list(
            dict.fromkeys(
                action.actor
                for action in actions
            )
        )

        actors = [
            ObservedActor(
                id=f"{segment_id}-actor-{index}",
                name=name,
            )
            for index, name in enumerate(
                actor_names,
                start=1,
            )
        ]

        actor_ids = {
            self._name_key(actor.name): actor.id
            for actor in actors
        }

        return actors, actor_ids

    def _build_objects(
        self,
        segment_id: str,
        extracted_objects: list[ExtractedObject],
    ) -> tuple[list[ObservedObject], dict[str, str]]:
        objects = [
            ObservedObject(
                id=f"{segment_id}-object-{index}",
                name=item.name,
                object_type=item.object_type,
            )
            for index, item in enumerate(
                extracted_objects,
                start=1,
            )
        ]

        object_ids = {
            self._name_key(item.name): item.id
            for item in objects
        }

        return objects, object_ids

    def _build_actions(
        self,
        *,
        segment_id: str,
        extracted_actions: list[ExtractedAction],
        actor_ids: dict[str, str],
        object_ids: dict[str, str],
        include_timestamps: bool,
    ) -> list[ObservedAction]:
        actions: list[ObservedAction] = []

        for index, item in enumerate(
            extracted_actions,
            start=1,
        ):
            start_time_ms: int | None = None
            end_time_ms: int | None = None

            if include_timestamps:
                if (
                    item.start_time is None
                    or item.end_time is None
                ):
                    raise ValueError(
                        "Video actions must contain start and end timestamps"
                    )

                start_time_ms = self._timestamp_to_ms(
                    item.start_time
                )

                end_time_ms = self._timestamp_to_ms(
                    item.end_time
                )

            actions.append(
                ObservedAction(
                    id=f"{segment_id}-action-{index}",
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                    actor_id=self._lookup(
                        actor_ids,
                        item.actor,
                        "actor",
                    ),
                    action=item.action,
                    object_id=self._optional_lookup(
                        object_ids,
                        item.object_name,
                        "object",
                    ),
                    instrument_id=self._optional_lookup(
                        object_ids,
                        item.instrument_name,
                        "instrument",
                    ),
                    target_id=self._optional_lookup(
                        object_ids,
                        item.target_name,
                        "target",
                    ),
                )
            )

        return actions

    def _lookup(
        self,
        values: dict[str, str],
        name: str,
        kind: str,
    ) -> str:
        identifier = values.get(
            self._name_key(name)
        )

        if identifier is None:
            raise ValueError(
                f"Unknown {kind}: {name!r}"
            )

        return identifier

    def _optional_lookup(
        self,
        values: dict[str, str],
        name: str | None,
        kind: str,
    ) -> str | None:
        if name is None:
            return None

        return self._lookup(
            values,
            name,
            kind,
        )

    @staticmethod
    def _action_prompt(
        *,
        require_timestamps: bool,
    ) -> str:
        time_instruction = (
            "Copy the start and end timestamps exactly. "
            if require_timestamps
            else "Do not create timestamps. "
        )

        return (
            "Extract exactly one observed action. "
            f"{time_instruction}"
            "Return the actor and a short action verb. "
            "If the actor is not stated, use unspecified actor. "
            "object_name is the item directly acted on. "
            "instrument_name is a tool used to perform the action. "
            "target_name is the destination or surface. "
            "When a worker picks up a screwdriver, the screwdriver "
            "is object_name, not instrument_name. When a worker "
            "tightens a screw using a screwdriver, the screw is "
            "object_name and the screwdriver is instrument_name. "
            "Use null when object_name, instrument_name, or "
            "target_name is absent. Return data only."
        )

    @staticmethod
    def _normalization_prompt() -> str:
        return (
            "Convert the free-form process description into structured "
            "segments. You must return at least one segment. Create one "
            "segment for each meaningful process step. Use only facts "
            "explicitly supported by the input. Do not invent actors, "
            "objects, tools, targets, actions, timestamps, or "
            "uncertainties. For each segment, provide a short scene "
            "summary, actions, all referenced objects, and explicit "
            "uncertainties. Use segment IDs in the form text-segment-1, "
            "text-segment-2, and so on. Repeat the actor for every action. "
            "Never return a null actor. Every non-null object_name, "
            "instrument_name, and target_name must appear in the same "
            "segment's objects list. Do not create timestamps. "
            "Return data only."
        )

    @staticmethod
    def _normalization_retry_prompt() -> str:
        return (
            "Convert the process description into a structured text "
            "instruction. You must return at least one segment. Do not "
            "return an empty segments list. Put all actions in one segment "
            "if you are unsure how to divide the process. Use only facts "
            "from the input. Every segment must have a non-empty segment_id, "
            "scene, actions list, and objects list. Repeat the actor for "
            "every action and never return a null actor. Every non-null "
            "object_name, instrument_name, and target_name must appear in "
            "the same segment's objects list. Do not create timestamps. "
            "Return data only."
        )