from __future__ import annotations

import os
from typing import Literal, TYPE_CHECKING
from collections import deque
from dataclasses import dataclass

import isaacus

if TYPE_CHECKING:
    from isaacus.types.ilgs.v1.document import Document as ILGSDocument
    from isaacus.types.ilgs.v1.span import Span
    from isaacus.types.ilgs.v1.segment import Segment
    
POSSIBLE_ANNOTATIONS = ( 
    "heading",
    "title_heading",            # Reserved for document title. 
    "title",
    "cross_ref",                # Cross referencing another annotation
    "junk",
    "quote",                    
    "ext_ref",                  # External reference 
    "src_ref"                   # Pointed to by a cross_ref
)

@dataclass
class _Annotation: 
    start: int                      # Annotation starting index
    end  : int                      # Annotation Ending index 
    kind : Literal[
        "heading",
        "title_heading",
        "title", 
        "cross_ref",
        "junk",
        "quote",
        "ext_ref",
        "src_ref",
    ]
    # kind=="heading" only 
    level: int | None = None        # heading level; only relevant for kind == heading
    seg_id: int | None = None       # Segment ID the heading is a part of 

    # kind=="cross_ref" or "src_ref" only 
    start_id: int | None = None     # Starting segment ID of reference (where the cross_ref will point to)


def text_to_markdown(
    text: str | ILGSDocument,
    *,
    cross_references: bool = True,
    strike_junk: bool = True,
    wrap_quotes: bool = True,
    italicise_ext_refs: bool = True,
    isaacus_client: isaacus.Isaacus | None = None,
) -> str:
    """Converts plain text to markdown.

    Args:
        text (str | ILGSDocument): Input text to be converted to markdown.

        cross_references (bool, optional): Whether or not text referencing other entities in the document should contain links to those entities.

        strike_junk (bool, optional): Whether or not to cross out junk text (headers, footings, etc.).

        wrap_quotes (bool, optional): Whether or not to turn non-inline quotes into markdown block quotes.

        italicise_ext_refs (bool, optional): Whether or not to italicise any external references.

        isaacus_client (isaacus.Isaacus, optional): A pre-initialised instance of an isaacus API client. If `None`, a new instance will be created instead.
    """

    # If input is raw text, convert into ILGS document
    if isinstance(text, str):
        if isaacus_client is None:
            api_key = os.getenv("ISAACUS_API_KEY")
            if api_key is None:
                raise ValueError(
                    """ Could not find an Isaacus API key in environment variables. See https://platform.isaacus.com/accounts/signup/."""
                )

            isaacus_client = isaacus.Isaacus()

        ilgs_doc = (
            isaacus_client.enrichments.create(
                model="kanon-2-enricher",
                texts=text,
                overflow_strategy="auto",
            )
            .results[0]
            .document
        )
    else:
        ilgs_doc = text
    text = ilgs_doc.text
    
    # Idea: Gather all annotations to queue, build a hierarchy of events ordered by index,
    # then perform the necessary plain text -> markdown transformations 
    # as we iterate over the input text
    ann_queue: list[_Annotation] = []
    headings = deque(sorted(ilgs_doc.headings, key=lambda span: span.start))
    segs = sorted(ilgs_doc.segments, key=lambda s: (s.span.start, -s.span.end))
    num_segs = len(segs)

    # We need the disjoint span ranges of each segment for heading->segment mapping 
    # to ensure (or increase the likelihood of) headings being unique to segments which actually have a heading
    # rather than just ones that contain it. 
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
    if (title := ilgs_doc.title) is not None:
        if title.start <= headings[0].start < title.end:
            h = headings.popleft()
            ann_queue.append(_Annotation(h.start, h.end, kind="title_heading"))

    # If we want cross_references, then we benefit from having a segment id -> span map 
    seg_id_to_span: dict[str, Span] = {} if cross_references else None
    id_to_seg: dict[str, Segment] = {None: None}
    has_heading: set[tuple[int, int]] = set()

    # Find headings and add their annotations with levels
    for idx, seg in enumerate(segs):
        if (seg.span.start, seg.span.end) in has_heading:
            continue

        if seg_id_to_span is not None: 
            seg_id_to_span[seg.id] = seg.span
        
        id_to_seg[seg.id] = seg
        level = seg.level

        span_start, span_end = disjoint_seg_spans[num_segs-idx-1]
        if span_end - span_start <= 0:
            continue

        while headings and headings[0].start < span_start:
            headings.popleft()

        # Check if there's a heading in our disjointified span interval
        if headings and span_start <= headings[0].start < span_end:
            kind = "heading"
            h = headings.popleft() 
            ann_start, ann_end = h.start, h.end

        # ===== Don't use seg.title anymore ===== 
       # elif seg.title is not None or seg.code is not None:
       #     #  fallback; we can't use a heading, so see if we can string together a title 
       #     kind = "title"
       #     ordered_parts = (seg.code, seg.title)
       #     ann_start = None
       #     for idx, part in enumerate(ordered_parts):
       #         if part is None:
       #             continue 
       #         
       #         ann_end = part.end
       #         if ann_start is None:
       #             ann_start = part.start     
             
        else:
            continue

        # ensure heading depth is with respect to parents with headings
        curr = id_to_seg[seg.parent]
        while curr is not None:
            # decrement level for each parent segment missing a title
            if (curr.span.start, curr.span.end) not in has_heading:
                level -= 1
            curr = id_to_seg[curr.parent]
        
        has_heading.add((seg.span.start, seg.span.end))
        ann_queue.append(_Annotation(ann_start, ann_end, kind=kind, level=level))

    # Gather annotations for the optional parameters!
    optional_annotators = {
        "cross_ref": (ilgs_doc.crossreferences, cross_references),
        "junk": (ilgs_doc.junk, strike_junk),
        "quote": (ilgs_doc.quotes, wrap_quotes),
        "ext_ref": (ilgs_doc.external_documents, italicise_ext_refs) 
    }
    for kind, (annotators, param) in optional_annotators.items():
        if not param: 
            continue
        for ann in annotators:
            match kind:
                case "cross_ref":
                    start_id = ann.start # references' start segment id
                    # Add annotations for the text itself (indicated by ann.span)
                    ann_queue.append(
                        _Annotation(ann.span.start, ann.span.end, kind=kind, start_id=start_id)
                    )

                    # need to add in annotations for the source reference as well, for anchoring
                    start_seg_span = seg_id_to_span[start_id]
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
    tie_break = {
        "heading": 0, 
        **{ann_kind: 1 for ann_kind in POSSIBLE_ANNOTATIONS if ann_kind != "heading"}
    }

    # We have all our annotations, add them to event queue, sort by index
    events = []
    for ann in ann_queue:
        events.append((ann.start, "start", ann))
        events.append((ann.end, "end", ann))
    events.sort(key=lambda a: (a[0], tie_break[a[-1].kind]))
    
    in_heading = True
    prev_kind = ""
    curr_idx = 0
    md: list[str] = [] # Output markdown
    for pos, t, ann in events:
        kind = ann.kind
        
        # Headings may span multiple lines; in this case, we want to concatenate them 
        # onto the same line
        if in_heading and kind == "heading" or kind == "title":
            # stich heading together
            pieces = [s.strip() for s in text[curr_idx:pos].split()]
            if pieces:
                md.append(" ".join(pieces)+"\n")
        elif pos != curr_idx:
            md.append(text[curr_idx:pos])

        match ann.kind:
            case "heading" | "title":
                # prepend with # based on level (at least 2)
                if t=="start":
                    md.append(f"\n{'#'*(min(6, ann.level+2))} ")
                    in_heading = True
                if t=="end":
                    in_heading = False

            case "title_heading":
                # prepend single # 
                md.append("# " if t=="start" else "\n")
                in_heading = t=="start"

            case "cross_ref":
                # Set hyperlink 
                newlines = 0
                # Ensure that we remove all added whitespace/newlines before enclosing in brackets
                if md and t=="end":
                    original_len = len(md[-1])
                    md[-1] = md[-1].rstrip()
                    newlines = original_len - len(md[-1])

                appended = "\n" * newlines
                md.append("[" if t=="start" else f"](#{ann.start_id.replace(':', '-')}){appended}")

            case "junk":
                # Cross out junk
                md.append("~~")
                
            case "quote":
                # turn into blockquotes only if quote appears on newline
                if t == "start" and pos > 0 and text[pos-1] == "\n":
                    md.append("> ")
                else: 
                    md.append("\n\n")

            case "ext_ref":
                # Italicise external references
                md.append("*")
            
            case "src_ref":
                # set anchor at source
                if t == "start" and prev_kind != "src_ref": 
                    tag = f"""<a id="{ann.start_id.replace(":", "-")}"></a>"""
                    md.append(tag)         

        prev_kind = ann.kind
        curr_idx = pos
        
    md.append(text[curr_idx:])

    # Finally, need to standardise formatting w.r.t. newlines: Don't want more than two blank lines in a row
    raw = "".join(md)
    
    # Headings and obvious pre-existing lists should have exactly one
    # blank line before and after 
    list_like = ["-", "•", "➤"]
    clean = []
    for line in raw.splitlines(True):
        is_item = any(line.strip().startswith(c) for c in list_like)
        prev_is_item = clean and any(clean[-1].strip().startswith(c) for c in list_like)
        heading = line.startswith("#")

        if not heading and not prev_is_item and not is_item:
            clean.append(line)
            continue

        if heading: 
            clean.append(f"\n{line}\n")
        elif is_item or prev_is_item:
            clean.append(f"\n{line}")
 
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
