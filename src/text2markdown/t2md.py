from __future__ import annotations

import os
from typing import Literal, TYPE_CHECKING
from collections import deque
from dataclasses import dataclass

if TYPE_CHECKING:
    import isaacus
    from isaacus.types.ilgs.v1.document import Document as ILGSDocument
    from isaacus.types.ilgs.v1.span import Span
    from isaacus.types.ilgs.v1.segment import Segment
    
    
POSSIBLE_ANNOTATIONS = ( 
    "heading",
    "title_heading",            # Reserved for document title. 
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
        "cross_ref",
        "junk",
        "quote",
        "ext_ref",
        "src_ref"
    ]
    # kind=="heading" only 
    level: int | None = None        # heading level; only relevant for kind == heading
    seg_id: int | None = None       # Segment ID the heading is a part of 

    # kind=="cross_ref" or "src_ref" only 
    start_id: int | None = None     # Starting segment ID of reference (where the cross_ref will point to)


def has_title(seg: Segment):
    return not (seg.title is None and seg.code is None and seg.type_name is None)


def text_to_markdown(
    text: str | ILGSDocument,
    *,
    cross_references: bool = True,
    strike_junk: bool = True,
    wrap_quotes: bool = True,
    italicise_ext_refs: bool = True,
    isaacus_client: "isaacus.Isaacus | None" = None,
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
            try:
                import isaacus

            except ImportError as e:
                raise ImportError(
                    """ The Isaacus package is required if an ILGSDocument is not provided. You can install this with `pip install isaacus`."""
                ) from e

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
    segs = deque(sorted(ilgs_doc.segments, key=lambda s: (s.span.start, -s.span.end)))
    # Check for title
    if (title := ilgs_doc.title) is not None:
        ann_queue.append(_Annotation(title.start, title.end, kind="title_heading"))

    # If we want cross_references, then we benefit from having a segment id -> span map 
    seg_id_to_span: dict[str, Span] = {} if cross_references else None
    id_to_seg: dict[str, Segment] = {None: None}

    # Find headings and add their annotations with levels
    for seg in segs:
        if seg_id_to_span is not None: 
            seg_id_to_span[seg.id] = seg.span

        id_to_seg[seg.id] = seg

        while headings and headings[0].start < seg.span.start:
            headings.popleft()

        if headings and seg.span.start <= headings[0].start < seg.span.end and has_title(seg):
            h = headings.popleft()
            
            # Ensure depth level is relative to other titled segments
            level = seg.level
            curr = id_to_seg[seg.parent]
            while curr is not None:
                # decrement level for each parent segment missing a title
                if not has_title(curr):
                    level -= 1

                curr = id_to_seg[curr.parent]
            ann_queue.append(_Annotation(h.start, h.end, kind="heading", level=level))

    # Gather annotations for the enabled optional parameters
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
    events = []
    for ann in ann_queue:
        events.append((ann.start, "start", ann))
        events.append((ann.end, "end", ann))
    events.sort(key=lambda a: (a[0], tie_break[a[-1].kind]))

    prev_kind = ""
    curr_idx = 0
    md: list[str] = [] # Output markdown
    for pos, t, ann in events:
        kind = ann.kind
        md.append(text[curr_idx:pos])

        match ann.kind:
            case "heading":
                level = ann.level
                # Ensure headers have empty lines above and below for consistent rendering
                if t=="start":
                    md.append(f"{'#'*(level+2)} ")

            case "title_heading":
                md.append("# " if t=="start" else "\n")

            case "cross_ref":
                md.append("[" if t=="start" else f"](#{ann.start_id.replace(":", "-")})")

            case "junk":
                # Cross out junk, add extra white-space at the end
                md.append("~~" + " "*(t=="end"))
                
            case "quote":
                # iffy check for whether or not the quote is on a newline
                if t == "start" and pos > 0 and text[pos-1] == "\n":
                    md.append("> ")
                else: 
                    md.append("\n")

            case "ext_ref":
                # Italicise external references
                md.append("*" + " "*(t=="end"))
            
            case "src_ref":
                # set anchor at source
                if t == "start" and prev_kind != "src_ref": 
                    tag = f"""<a id="{ann.start_id.replace(":", "-")}"></a>"""
                    md.append(tag)         
        prev_kind = ann.kind
        curr_idx = pos
        
    md.append(text[curr_idx:])

    # Finally, need to standardise formatting w.r.t. newlines
    raw = "".join(md)
    newline_removed = (f"{line}\n" for line in raw.splitlines() if line.strip())

    # Headings and obvious pre-existing lists should have exactly one
    # blank line before and after 
    list_like = ["-", "•"]
    clean = []
    for line in newline_removed:
        is_item = any(line.strip().startswith(c) for c in list_like)
        prev_is_item = clean and any(clean[-1].strip().startswith(c) for c in list_like)
        heading = line.startswith("#")

        if not heading and not is_item and not prev_is_item:
            clean.append(f"{line.rstrip()}<br>\n")
            continue
        
        # Adding newlines after headings; check if blank spaces already exist first
        if heading:
            if clean and clean[-1].endswith("\n\n"):
                clean.append(f"{line}\n")
            else:
                clean.append(f"\n{line}\n")
            continue
        
        # Adding newlines between list blocks, only if there isn't a pre-existing blank space
        if (clean and not clean[-1].endswith("\n\n")) and (is_item and not prev_is_item):
            clean.append(f"\n{line}")
        elif prev_is_item and not is_item:
            clean.append(f"\n{line}")
        else:
            clean.append(line)
        

    return "".join(clean).strip()
