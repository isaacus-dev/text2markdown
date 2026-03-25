from __future__ import annotations

import re

from typing import Literal, Iterable, NamedTuple
from collections import deque
from dataclasses import dataclass

import isaacus

from isaacus.types.ilgs.v1.segment import Segment
from isaacus.types.ilgs.v1.document import Document as ILGSDocument

_LIST_PATTERNS = [
    re.compile(r"^\s{0,3}[-+*]\s+"),  # Unordered lists: -, *, +
    re.compile(r"^\s{0,3}\d+\.\s+"),  # Ordered lists: 1.  2.  10.
    re.compile(r"^\s{0,3}\d+\)\s+"),  # Ordered lists with parentheses: 1)  2)
]


@dataclass
class _Annotation:
    start: int  # Annotation starting index
    end: int  # Annotation ending index
    kind: Literal[
        "heading",
        "subtitle",
        "xref",  # Cross referencing another annotation
        "junk",
        "quote",
        "ext_ref",  # External references
        "src_ref",  # Pointed to by a xref
    ]
    force_blank_line: bool = False
    level: int | None = None  # kind=="heading" only
    start_id: int | None = None  # kind=="xref" or "src_ref" only

    _static_tags = {  # Markdown tags to attach to each `_Annotation` kind
        "subtitle": ("""<p style="text-align: center;">""", "</p>"),
        "junk": ("~~", "~~"),
        "quote": ("> ", None),
        "ext_ref": ("*", "*"),  # Italicise external references
    }

    @property
    def tags(self) -> tuple[str, str | None]:
        """Returns the markdown/html tags that need to be added at the `start` and `end` index of this `_Annotation`, respectively."""
        match self.kind:
            case "heading":
                if self.level == 1:
                    return ("""<h1 style="text-align: center;">""", "</h1>")
                else:
                    return (f"\n{'#' * min(6, self.level)} ", None)

            case "xref":
                return ("[", f"](#{self.start_id.replace(':', '-')})")

            case "src_ref":
                return (f"""<a id="{self.start_id.replace(":", "-")}"></a>""", None)

        return self._static_tags[self.kind]

    def __hash__(self):
        return hash((self.start, self.end, self.kind, self.force_blank_line, self.level, self.start_id))


class _Event(NamedTuple):
    position: int
    time: Literal["start", "end"]
    annotation: _Annotation


# ==== START HELPER FUNCTIONS ====


def _is_list_line(line: str) -> bool:
    """Determines if `line` will be rendered as a list item in markdown."""
    return any(p.match(line) for p in _LIST_PATTERNS)


def _annotate_each_line(
    full_annotation: _Annotation, doc_text: str, add_newlines: bool = False
) -> Iterable[_Annotation]:
    """Creates `_Annotation`s on `doc_text` for each line included in `full_annotation`."""
    a_start, a_end = full_annotation.start, full_annotation.end
    span_text_lines = doc_text[a_start:a_end].splitlines(keepends=True)
    offset = a_start
    for i, line in enumerate(span_text_lines):
        # add newline at the end of annotation group if `force_blank == True``
        add_newline = (i == len(span_text_lines) - 1) and add_newlines

        line_start = offset
        line_end = offset + len(line)

        # skip whitespace lines
        if line.strip():
            yield _Annotation(
                line_start,
                line_end,
                kind=full_annotation.kind,
                level=full_annotation.level,
                start_id=full_annotation.start_id,
                force_blank_line=add_newline,
            )

        offset = line_end


def _safe_append_tag(md: list[str], tag: str | None):
    """Safely appends `tag` to the last non-newline/whitespace entry of `md`, preserving
    trailing and leading newlines/whitespaces.
    """
    if tag is None:
        return

    i = len(md) - 1
    while i > 0 and not md[i].strip():
        i -= 1

    text_to_tag = md[i]
    stripped = text_to_tag.rstrip()
    md[i] = text_to_tag[: len(stripped)] + tag + text_to_tag[len(stripped) :]


def _filter_events(events: list[_Event]) -> list[_Event]:
    """Filters `events`, removing overlapping annotations which could break the markdown output."""
    priority = {
        "junk": 0,  # Lower value = lower priority
        "ext_ref": 1,
        "subtitle": 2,
        "xref": 3,
    }
    active: list[_Annotation] = []  # stack of active annotations
    filtered_events: list[_Event] = []

    for e in events:
        ann = e.annotation
        kind = ann.kind
        if kind not in priority.keys():
            filtered_events.append(e)
            continue

        if e.time == "start":
            # Check conflict with currently active annotations
            to_remove = []
            discard = False

            for a in active:
                # overlap condition
                if ann.start < a.end and ann.end > a.start:
                    if priority[kind] > priority[a.kind]:
                        to_remove.append(a)
                    else:
                        discard = True
                        break

            if discard:
                continue

            # Remove weaker overlapping annotations
            if to_remove:
                active = [a for a in active if a not in to_remove]
                filtered_events = [ev for ev in filtered_events if ev.annotation not in to_remove]

            active.append(ann)
            filtered_events.append(e)

        else:  # time == end
            # only append if the start has already been seen
            if ann in active:
                active.remove(ann)
                filtered_events.append(e)

    # replace events with filtered version
    return filtered_events


# ==== END HELPER FUNCTIONS ====


def text2markdown(
    text: str | ILGSDocument,
    *,
    link_xrefs: bool = True,
    strike_junk: bool = True,
    block_quotes: bool = True,
    italicize_refs: bool = True,
    enrichment_model: str = "kanon-2-enricher",
    isaacus_client: isaacus.Isaacus | None = None,
) -> str:
    """Intelligently converts plain text into Markdown.

    Args:
        text (str | ILGSDocument): Input to be converted into Markdown. If an Isaacus Legal Graph Schema (ILGS) Document is supplied, this function will convert the Document's text into Markdown without needing to enrich it first with an Isaacus enrichment model.

        link_xrefs (bool, optional): Whether to link cross-references in the input text to their targets, for example, linking "as mentioned in Section 2.1" to the relevant section.

        strike_junk (bool, optional): Whether to strike out junk text.

        block_quotes (bool, optional): Whether to transform non-inline quotes into Markdown block quotes.

        italicize_refs (bool, optional): Whether to italicize the names of any referenced documents, for example, "as mentioned in *Smith v. Jones*".

        enrichment_model (str, optional): The name of the Isaacus enrichment model to use for converting the input text into Markdown. Defaults to the latest and most advanced Isaacus enrichment model, currently `kanon-2-enricher`.

        isaacus_client (isaacus.Isaacus, optional): An Isaacus API client to use for enriching the input text with an Isaacus enrichment model if the input is not already an Isaacus Legal Graph Schema (ILGS) Document. If `None`, a new instance will be created instead where necessary.
    """

    # Convert the input text into an Isaacus Legal Graph Schema (ILGS) Document if it is not one already.
    if isinstance(text, str):
        if isaacus_client is None:
            isaacus_client = isaacus.Isaacus()

        response = isaacus_client.enrichments.create(model=enrichment_model, texts=text, overflow_strategy="auto")
        doc = response.results[0].document

    else:
        doc = text

    text = doc.text

    # Idea: Gather all annotations to queue, build a hierarchy of events ordered by index,
    # then perform the necessary plain text -> markdown transformations
    # as we iterate over the input text
    ann_queue: set[_Annotation] = set()
    headings = deque(sorted([h for h in doc.headings if h.decode(text).strip()], key=lambda span: span.start))
    segs = sorted(doc.segments, key=lambda s: (s.span.start, -s.span.end))
    num_segs = len(segs)

    # we want to 'disjointify' our span segments. If we have segment spans [[25, 40], [30, 50]],
    # then it is desirable to have a representation in the form [[25, 30], [30, 50]]. If we have it in this form,
    # we can say the heading [30, 40] belongs to the segment [30, 50] because it is uniquely contained in it
    # in the disjoint representation
    disjoint_seg_spans: list[tuple[int, int]] = []
    for seg in reversed(segs):
        dj_start = seg.span.start
        if disjoint_seg_spans and seg.span.end >= disjoint_seg_spans[-1][0]:
            # this segment ends after the start of the next segment; cut off the intersection
            dj_end = disjoint_seg_spans[-1][0]
        else:
            dj_end = seg.span.end

        disjoint_seg_spans.append((dj_start, dj_end))

    # Check for title; level 1 heading "#" is reserved for the title heading
    if (title := doc.title) and headings and headings[0].start <= title.start < headings[0].end:
        h = headings.popleft()
        ann_queue.add(_Annotation(h.start, h.end, kind="heading", level=1))

    # Extract subtitle
    if (subtitle := doc.subtitle) and headings and headings[0].start <= subtitle.start < headings[0].end:
        h = headings.popleft()
        ann_queue.add(_Annotation(h.start, h.end, kind="subtitle"))

    id_to_seg: dict[str | None, Segment | None] = {None: None}
    has_heading: set[tuple[int, int]] = set()

    # Find headings and add their annotations with levels
    for idx, seg in enumerate(segs):
        id_to_seg[seg.id] = seg

        span_start, span_end = disjoint_seg_spans[num_segs - idx - 1]  # disjoint span interval
        if span_end - span_start <= 0:
            continue

        curr_level = seg.level + 2  # offset counting to start from 2 instead of 0 (number of #'s in markdown format)
        while headings and headings[0].start < span_start:
            h = headings.popleft()
            # Default segmentless headings' level
            ann_queue.add(_Annotation(h.start, h.end, kind="heading", level=curr_level))

        annotations: list[tuple[int, int, int]] = []
        # annotate headings in segment
        lev = curr_level
        while headings and span_start <= headings[0].start < span_end:
            h = headings.popleft()
            annotations.append((h.start, h.end, lev))
            lev += 1

        if not annotations:
            # no heading in this segment
            continue

        for ann in annotations:
            ann_start, ann_end, ann_level = ann

            # ensure heading depth is with respect to parents with headings
            curr = id_to_seg[seg.parent]
            while curr is not None:
                # decrement level for each parent segment missing a heading
                if (curr.span.start, curr.span.end) not in has_heading:
                    ann_level -= 1
                curr = id_to_seg[curr.parent]
            ann_queue.update(
                _annotate_each_line(_Annotation(ann_start, ann_end, kind="heading", level=max(2, ann_level)), text)
            )

        has_heading.add((seg.span.start, seg.span.end))

    # Add any remaining headings which come after the last segment
    for heading in headings:
        ann_queue.add(_Annotation(heading.start, heading.end, kind="heading", level=2))

    # We've annotated all headings, now gather annotations for the optional parameters.
    optional_annotators = {
        "xref": (doc.crossreferences, link_xrefs),
        "junk": (doc.junk, strike_junk),
        "quote": (doc.quotes, block_quotes),
        "ext_ref": (doc.external_documents, italicize_refs),
    }
    for kind, (annotators, asked_to_implement) in optional_annotators.items():
        if not asked_to_implement:
            continue

        for ann in annotators:
            match kind:
                case "xref":
                    start_id = ann.start  # references' start segment id
                    # Add annotations for the text itself (indicated by ann.span)
                    ann_queue.update(
                        _annotate_each_line(
                            _Annotation(ann.span.start, ann.span.end, kind=kind, start_id=start_id), text
                        )
                    )

                    # need to add in annotations for the source reference as well, for anchoring
                    start_seg_span = id_to_seg[start_id].span
                    ann_queue.add(
                        _Annotation(start_seg_span.start, start_seg_span.end, kind="src_ref", start_id=start_id)
                    )

                case "junk":
                    ann_queue.update(_annotate_each_line(_Annotation(ann.start, ann.end, kind=kind), text))

                case "quote":
                    if ann.span.start > 0 and text[ann.span.start - 1] != "\n":
                        # Only annotate block quotes; must be preceded with '\n' char
                        continue
                    ann_queue.update(
                        _annotate_each_line(
                            _Annotation(ann.span.start, ann.span.end, kind=kind), text, add_newlines=True
                        )
                    )

                case "ext_ref":
                    # Each external reference has an array of mentions we want to annotate.
                    for mention in ann.mentions:
                        ann_queue.update(_annotate_each_line(_Annotation(mention.start, mention.end, kind=kind), text))

    events: list[_Event] = []
    for ann in ann_queue:
        events.append(_Event(ann.start, "start", ann))
        # Don't need end events for some annotation types
        if ann.kind != "src_ref":
            events.append(_Event(ann.end, "end", ann))

    kind_priority = {"heading": 6, "quote": 5, "ext_ref": 4, "junk": 3, "xref": 2, "subtitle": 1, "src_ref": 0}
    zero_length_annotations = {"src_ref"}

    def event_sort_key(e: _Event):
        """Determines behaviour if two events occur at the same index."""
        kind, start, end = e.annotation.kind, e.annotation.start, e.annotation.end
        if e.time == "start":
            start_first = 1
            kind_order = -kind_priority[kind]
            length_order = -(end - start) if kind not in zero_length_annotations else 1

        else:
            start_first = 0
            kind_order = kind_priority[kind]
            length_order = end - start if kind not in zero_length_annotations else -1

        return (e.position, start_first, length_order, kind_order)

    events.sort(key=event_sort_key)
    events = _filter_events(events)

    # ===== Process events =====
    md: list[str] = []  # Output markdown
    curr_idx = 0
    for pos, t, ann in events:
        kind = ann.kind
        if curr_idx != pos:
            md.append(text[curr_idx:pos])

        if t == "start":
            md.append(ann.tags[0])

        else:
            _safe_append_tag(md, ann.tags[1])
            if ann.force_blank_line:
                md.append("\n\n")

        curr_idx = pos

    md.append(text[curr_idx:])
    raw = "".join(md)

    # We have some post-processing to do
    newlines_added = (f"{line}\n" if line.startswith("#") else line for line in raw.splitlines(True))

    # ensure every line in the output is surrounded by exactly one blank line before and after,
    # except for quotations. Additionally, preserve indentation by using html tags.
    prev_is_blank = False
    blank_removed: list[str] = []
    for line in "".join(newlines_added).splitlines():
        if prev_is_blank and not line.strip():
            # second blank in a row
            continue
        prev_is_blank = not line.strip()

        # prevent markdown list rendering
        if _is_list_line(line) and line.lstrip() == line:
            line = f"&#8203;{line}"

        # Convert leading tabs/whitespace to html indent flags
        line = line.expandtabs(4)
        line = re.sub(r"^(?:\s{4})+", lambda m: "&emsp;" * (len(m.group(0)) // 4), line)
        line = re.sub(r"^((?:&emsp;)*)\s{2}", r"\1&ensp;", line)
        line = re.sub(r"^((?:&emsp;|&ensp;)*)\s", r"\1&nbsp;", line)

        if not line.startswith("> "):
            line = line.rstrip("\n") + "\n"
            prev_is_blank = True

        blank_removed.append(line + "\n" if line.strip() else line)

    return "".join(blank_removed).strip()
