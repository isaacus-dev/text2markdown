from __future__ import annotations

from typing import Literal
from collections import deque
from dataclasses import dataclass

import isaacus

from isaacus.types.ilgs.v1.segment import Segment
from isaacus.types.ilgs.v1.document import Document as ILGSDocument

_POSSIBLE_ANNOTATIONS = (
    "heading",
    "title_heading",  # Reserved for document title.
    "cross_ref",  # Cross referencing another annotation
    "junk",
    "quote",
    "ext_ref",  # External reference
    "src_ref",  # Pointed to by a cross_ref
)


@dataclass
class _Annotation:
    start: int  # Annotation starting index
    end: int  # Annotation Ending index
    kind: Literal[
        "heading",
        "title_heading",
        "cross_ref",
        "junk",
        "quote",
        "ext_ref",
        "src_ref",
    ]
    # kind=="heading" only
    level: int | None = None  # heading level; only relevant for kind == heading
    seg_id: int | None = None  # Segment ID the heading is a part of

    # kind=="cross_ref" or "src_ref" only
    start_id: int | None = None  # Starting segment ID of reference (where the cross_ref will point to)


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

        link_cross_references (bool, optional): Whether to link cross-references in the input text to their targets, for example, linking "as mentioned in Section 2.1" to the relevant section.

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
    ann_queue: list[_Annotation] = []
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

    # Check for title
    if (title := doc.title) and title.start <= headings[0].start < title.end:
        h = headings.popleft()
        ann_queue.append(_Annotation(h.start, h.end, kind="title_heading"))

    id_to_seg: dict[str, Segment] = {None: None}
    has_heading: set[tuple[int, int]] = set()

    # Find headings and add their annotations with levels
    for idx, seg in enumerate(segs):
        id_to_seg[seg.id] = seg

        span_start, span_end = disjoint_seg_spans[num_segs - idx - 1]
        if span_end - span_start <= 0:
            continue

        while headings and headings[0].start < span_start:
            h = headings.popleft()
            # Default "segmentless" headings' level to current segment level
            ann_queue.append(_Annotation(h.start, h.end, kind="heading", level=seg.level))

        annotations: list[tuple[int, int, int]] = []
        level = seg.level
        # annotate any headings in our disjointified span interval
        while headings and span_start <= headings[0].start < span_end:
            h = headings.popleft()
            annotations.append((h.start, h.end, level))
            level += 1

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

            ann_queue.append(_Annotation(ann_start, ann_end, kind="heading", level=ann_level))

        has_heading.add((seg.span.start, seg.span.end))

    # Gather annotations for the optional parameters!
    optional_annotators = {
        "cross_ref": (doc.crossreferences, link_xrefs),
        "junk": (doc.junk, strike_junk),
        "quote": (doc.quotes, block_quotes),
        "ext_ref": (doc.external_documents, italicize_refs),
    }
    for kind, (annotators, param) in optional_annotators.items():
        if not param:
            continue
        for ann in annotators:
            match kind:
                case "cross_ref":
                    start_id = ann.start  # references' start segment id
                    # Add annotations for the text itself (indicated by ann.span)
                    ann_queue.append(_Annotation(ann.span.start, ann.span.end, kind=kind, start_id=start_id))

                    # need to add in annotations for the source reference as well, for anchoring
                    start_seg_span = id_to_seg[start_id].span
                    ann_queue.append(
                        _Annotation(start_seg_span.start, start_seg_span.end, kind="src_ref", start_id=start_id)
                    )

                case "junk":
                    ann_queue.append(_Annotation(ann.start, ann.end, kind=kind))

                case "quote":
                    ann_queue.append(_Annotation(ann.span.start, ann.span.end, kind=kind))

                case "ext_ref":
                    # Each external reference has an array of mentions we want to annotate.
                    for mention in ann.mentions:
                        ann_queue.append(_Annotation(mention.start, mention.end, kind=kind))

    # if two annotations occur at the same index, which action should execute first?
    tie_break = {"heading": 0, **{ann_kind: 1 for ann_kind in _POSSIBLE_ANNOTATIONS if ann_kind != "heading"}}

    # We have all our annotations, add them to event queue, sort by index
    events = []
    for ann in ann_queue:
        events.append((ann.start, "start", ann))
        events.append((ann.end, "end", ann))
    events.sort(key=lambda a: (a[0], tie_break[a[-1].kind]))

    in_heading = True
    curr_idx = 0
    md: list[str] = []  # Output markdown
    for pos, t, ann in events:
        kind = ann.kind

        # Headings may span multiple lines; in this case, we want to concatenate them
        # onto the same line
        if in_heading and kind == "heading":
            # stich heading together
            pieces = [s.strip() for s in text[curr_idx:pos].split()]
            if pieces:
                md.append(" ".join(pieces) + "\n")
        elif pos != curr_idx:
            md.append(text[curr_idx:pos])

        match ann.kind:
            case "heading":
                # prepend with # based on level (at least 2)
                if t == "start":
                    md.append(f"\n{'#' * (min(6, ann.level + 2))} ")
                    in_heading = True
                if t == "end":
                    in_heading = False

            case "title_heading":
                # prepend single #
                md.append("# " if t == "start" else "\n")
                in_heading = t == "start"

            case "cross_ref":
                # Set hyperlink
                newlines = 0
                # Ensure that we remove all added whitespace/newlines before enclosing in brackets
                if md and t == "end":
                    original_len = len(md[-1])
                    md[-1] = md[-1].rstrip()
                    newlines = original_len - len(md[-1])

                appended = "\n" * newlines
                md.append("[" if t == "start" else f"](#{ann.start_id.replace(':', '-')}){appended}")

            case "junk":
                # Cross out junk
                md.append("~~")

            case "quote":
                # turn into blockquotes only if quote appears on newline
                if t == "start" and pos > 0 and text[pos - 1] == "\n":
                    md.append("> ")
                else:
                    md.append("\n\n")

            case "ext_ref":
                # Italicise external references
                md.append("*")

            case "src_ref":
                # set anchor at source
                tag = f"""<a id="{ann.start_id.replace(":", "-")}"></a>"""
                if md and tag not in md[-1] and t == "start":
                    md.append(tag)

        curr_idx = pos

    md.append(text[curr_idx:])
    raw = "".join(md)

    # We want to ensure there are no more than two blank lines in a row
    # and we want headings to have blank lines before and after they're declared
    clean = (f"\n{line}\n" if line.startswith("#") else line for line in raw.splitlines(True))

    blank_lines = 0
    blank_removed: list[str] = []
    for line in "".join(clean).splitlines():
        if not line.strip():
            blank_lines += 1
        else:
            blank_lines = 0

        if blank_lines >= 2:
            continue

        if line.strip():
            blank_lines += 1
            blank_removed.append(f"{line.lstrip('.,: ')} \n\n")
        else:
            blank_removed.append(f"{line}\n")

    return "".join(blank_removed).strip()
