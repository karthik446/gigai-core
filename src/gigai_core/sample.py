"""One trial: one item asked once, with the spans that answered it."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import AwareDatetime

from .dataset import DatasetItem
from .models import CoreModel
from .providers import ToolCall, Turn
from .trace import CallError, ModelCall, ModelSpan, Span, ToolSpan
from .values import JsonValue, Outcome, ParseStatus


class Sample(CoreModel):
    """One trial of ``item_id`` inside a run: the conversation, the last output, and every span."""

    trace_id: str
    run_id: str
    item_id: str
    item_version: str
    cluster_id: str
    repeat_index: int
    input: tuple[Turn, ...]
    output_text: str
    output_parsed: JsonValue
    parse_status: ParseStatus
    outcome: Outcome
    error: CallError | None
    spans: tuple[Span, ...] = ()
    final_state: JsonValue = None
    recorded_at: AwareDatetime

    @property
    def model_spans(self) -> Sequence[ModelSpan]:
        """The model calls in order."""
        return tuple(span for span in self.spans if isinstance(span, ModelSpan))

    @property
    def tool_spans(self) -> Sequence[ToolSpan]:
        """The tool calls in order: what a path grader reads."""
        return tuple(span for span in self.spans if isinstance(span, ToolSpan))

    @property
    def tool_calls(self) -> Sequence[ToolCall]:
        """The tool calls of the sample, flattened from its tool spans."""
        return tuple(span.call for span in self.tool_spans)

    @property
    def last_call(self) -> ModelCall | None:
        """The last model call's facts."""
        spans = self.model_spans
        return spans[-1].call if spans else None

    def promoted(self, *, item_id: str, cluster_id: str, expected: JsonValue) -> DatasetItem:
        """This trial's conversation as a new dataset item; ``source_ids`` names the run and the trace it came from (N11).

        The user turns of ``input`` become the item's scripted ``turns``; its first system turn, if any, becomes
        ``system``.
        """
        system = next((turn.content for turn in self.input if turn.role == "system"), None)
        return DatasetItem(
            id=item_id,
            cluster_id=cluster_id,
            expected=expected,
            turns=tuple(turn for turn in self.input if turn.role == "user"),
            system=system,
            source_ids={"run_id": self.run_id, "trace_id": self.trace_id},
        )
