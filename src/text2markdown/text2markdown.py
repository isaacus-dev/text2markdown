from __future__ import annotations

import re

from typing import Literal
from collections import deque
from dataclasses import dataclass

import isaacus

from isaacus.types.ilgs.v1.segment import Segment
from isaacus.types.ilgs.v1.document import Document as ILGSDocument

_POSSIBLE_ANNOTATIONS = (
    "heading",
    "subtitle",
    "cross_ref",  # Cross referencing another annotation
    "junk",
    "quote",
    "ext_ref",  # External reference
    "src_ref",  # Pointed to by a cross_ref
)


@dataclass
class _Annotation:
    start: int  # Annotation starting index
    end: int  # Annotation ending index
    kind: Literal[
        "heading",
        "subtitle",
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
    start_id: int | None = None  # Starting segment ID of reference (where cross_ref points to)


def is_list_line(line: str):
    list_patterns = [
        # Unordered lists: -, *, +
        re.compile(r'^\s{0,3}[-+*]\s+'),

        # Ordered lists: 1.  2.  10.
        re.compile(r'^\s{0,3}\d+\.\s+'),

        # Ordered lists with parentheses: 1)  2)
        re.compile(r'^\s{0,3}\d+\)\s+'),
    ]
    return any(p.match(line) for p in list_patterns)


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

    # Check for title; level 1 heading "#" is reserved for the title heading
    if (title := doc.title) and headings[0].start <= title.start < headings[0].end:
        h = headings.popleft()
        ann_queue.append(_Annotation(h.start, h.end, kind="heading", level=1))

    # Extract subtitle
    if (subtitle := doc.subtitle) and headings[0].start <= subtitle.start < headings[0].end:
        h = headings.popleft()
        ann_queue.append(_Annotation(h.start, h.end, kind="subtitle"))

    id_to_seg: dict[str, Segment] = {None: None}
    has_heading: set[tuple[int, int]] = set()

    # Find headings and add their annotations with levels
    for idx, seg in enumerate(segs):
        id_to_seg[seg.id] = seg

        span_start, span_end = disjoint_seg_spans[num_segs - idx - 1] # disjoint span interval
        if span_end - span_start <= 0:
            continue

        curr_level = seg.level + 2 # offset counting to start from 2 instead of 0 (number of #'s in markdown format)
        while headings and headings[0].start < span_start: 
            h = headings.popleft()
            # Default "segmentless" headings' level to current segment level
            ann_queue.append(_Annotation(h.start, h.end, kind="heading", level=curr_level))

        annotations: list[tuple[int, int, int]] = []
        # annotate any headings we find in this segments disjointified span interval
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
                    ann_queue.append(_Annotation(start_seg_span.start, start_seg_span.end, kind="src_ref", start_id=start_id))

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

    anchors: set[str] = set()
    curr_idx = 0
    md: list[str] = []  # Output markdown
    for pos, t, ann in events:
        kind = ann.kind
        md.append(text[curr_idx:pos])
        match ann.kind:
            case "heading":
                # prepend with '#' based on level (at most 6)
                num_hashtags = min(6, ann.level)
                if num_hashtags == 1:
                    # Title heading, center it 
                    md.append("""<h1 style ="text-align: center;">""" if t == "start" else "</h1>")
                    curr_idx = pos
                    continue

                prefix = f"\n{'#' * num_hashtags} "
                if t == "start":
                    md.append(prefix)

                elif t == "end":
                    md.pop() # Avoid heading duplication
                    # add hashtags at the start of every heading line
                    pieces = [s for s in text[curr_idx:pos].split('\n') if s.strip()]
                    if pieces:
                        md.append(prefix.join(pieces))
            
            case "subtitle":
                md.append("""<p style="text-align: center;">""" if t == "start" else "</p>")

            case "cross_ref":
                md.append("[" if t == "start" else f"](#{ann.start_id.replace(':', '-')})")

            case "junk":
                # strike-out junk
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
                if tag not in anchors:
                    md.append(tag)
                anchors.add(tag)

        curr_idx = pos

    md.append(text[curr_idx:])
    raw = "".join(md)

    # We have some post-processing to do
    in_junk = False
    clean = []
    for line in raw.splitlines():
        # we're in a junk block; ensure '~~' is present on both ends of the line
        if in_junk:
            if not line.endswith("~~"):
                clean.append(f"\n~~{line}~~\n")
            else:
                clean.append(f"~~{line}\n")
                in_junk = False
            continue 

        if line.startswith("#"):
            # Ensure blank space before and after headings
            clean.append(f"\n{line}\n")

        elif line.startswith("~~") and not line.endswith("~~"):
            # Junk text which spans multiple lines
            clean.append(f"{line}~~\n")
            in_junk = True

        else:
            # Normal line
            clean.append(f"{line}\n")

    # ensure there are never consecutive blank lines in the output and render indents correctly
    blank_removed: list[str] = []
    for line in "".join(clean).splitlines():
        if not line.strip():
            continue 

        # Remove native list rendering for safe indent preservation
        if is_list_line(line) and line.lstrip() == line:
            line = f"&#8203;{line}"
        
        # Convert leading tabs/whitespace to html indent flags
        line = re.sub(r"^(?:\t|\s{4})+", lambda m: m.group(0).replace("\t", "&emsp;").replace(" "*4, "&emsp;"), line)
        line = re.sub(r"^((?:&emsp;)*)\s{2}", r"\1&ensp;", line)
        line = re.sub(r"^((?:&emsp;|&ensp;)*)\s", r"\1&nbsp;", line)
 
        blank_removed.append(f"{line}\n\n") 
    return "".join(blank_removed).strip()
