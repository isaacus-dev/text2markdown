from __future__ import annotations

import isaacus

from isaacus.types.ilgs.v1.document import Document as ILGSDocument

from .text2markdown import text2markdown


async def text2markdown_async(
    text: str | ILGSDocument,
    *,
    link_xrefs: bool = True,
    strike_junk: bool = True,
    block_quotes: bool = True,
    escape_lists: bool = True,
    italicize_refs: bool = True,
    italicize_terms: bool = True,
    enrichment_model: str = "kanon-2-enricher",
    isaacus_client: isaacus.AsyncIsaacus | None = None,
) -> str:
    """Intelligently converts plain text into Markdown asynchronously.

    Args:
        text (str | ILGSDocument): Input to be converted into Markdown. If an Isaacus Legal Graph Schema (ILGS) Document is supplied, this function will convert the Document's text into Markdown without needing to enrich it first with an Isaacus enrichment model.

        link_xrefs (bool, optional): Whether to link cross-references in the input text to their targets, for example, linking "as mentioned in Section 2.1" to the relevant section.

        strike_junk (bool, optional): Whether to strike out junk text.

        block_quotes (bool, optional): Whether to transform non-inline quotes into Markdown block quotes.

        escape_lists (bool, optional): Whether to escape list-like lines (lines starting with "-", "*", "+", or numbered lists).

        italicize_refs (bool, optional): Whether to italicize the names of any referenced documents, for example, "as mentioned in *Smith v. Jones*".

        italicize_terms (bool, optional): Whether to italicize the names of any defined terms.

        enrichment_model (str, optional): The name of the Isaacus enrichment model to use for converting the input text into Markdown. Defaults to the latest and most advanced Isaacus enrichment model, currently `kanon-2-enricher`.

        isaacus_client (isaacus.AsyncIsaacus, optional): An Isaacus API client to use for enriching the input text with an Isaacus enrichment model if the input is not already an Isaacus Legal Graph Schema (ILGS) Document. If `None`, a new instance will be created instead where necessary.
    """

    # Raise an error if supplied with a synchronous Isaacus client.
    if isinstance(isaacus_client, isaacus.Isaacus):
        raise ValueError("""\
        `text2markdown_async()` requires an asynchronous Isaacus client, but a synchronous Isaacus client was provided. Please supply an `isaacus.AsyncIsaacus` client or set `isaacus_client` to `None` to have an asynchronous client created automatically.""")

    # Convert the text into an Isaacus Legal Graph Schema (ILGS) Document if it is not one already.
    doc = text

    if isinstance(text, str):
        if isaacus_client is None:
            isaacus_client = isaacus.AsyncIsaacus()

        response = await isaacus_client.enrichments.create(model=enrichment_model, texts=text, overflow_strategy="auto")
        doc = response.results[0].document

    return text2markdown(
        doc,
        link_xrefs=link_xrefs,
        strike_junk=strike_junk,
        block_quotes=block_quotes,
        escape_lists=escape_lists,
        italicize_refs=italicize_refs,
        italicize_terms=italicize_terms,
        enrichment_model=enrichment_model,
        isaacus_client=None,
    )
