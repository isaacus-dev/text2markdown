""" Async version of t2md. """
from __future__ import annotations

from typing import TYPE_CHECKING
import os

from .t2md import text_to_markdown

from isaacus import AsyncIsaacus
from isaacus import Isaacus

if TYPE_CHECKING:
    from isaacus.types.ilgs.v1.document import Document as ILGSDocument


async def text_to_markdown_async(
    text: str | ILGSDocument,
    *,
    cross_references: bool = True,
    strike_junk: bool = True,
    wrap_quotes: bool = True,
    italicise_ext_refs: bool = True,
    isaacus_client: Isaacus | AsyncIsaacus | None = None,
) -> str:
    """Converts plain text to markdown with async support. If a synchronous Isaacus client or ILGS Document is provided, this function is identical to its synchronous counterpart.  

    Args:
        text (str | ILGSDocument): Input text to be converted to markdown.

        cross_references (bool, optional): Whether or not text referencing other entities in the document should contain links to those entities.

        strike_junk (bool, optional): Whether or not to cross out junk text (headers, footings, etc.).

        wrap_quotes (bool, optional): Whether or not to turn non-inline quotes into markdown block quotes.

        italicise_ext_refs (bool, optional): Whether or not to italicise any external references.

        isaacus_client (Isaacus, AsyncIsaacus, optional): A pre-initialised instance of an isaacus API client. If `None`, a new instance will be created instead.  
    """
    ilgs_doc = text
    if isinstance(text, str):
        if isaacus_client is None:
            api_key = os.getenv("ISAACUS_API_KEY")
            if api_key is None:
                raise ValueError(
                    """ Could not find an Isaacus API key in environment variables. See https://platform.isaacus.com/accounts/signup/."""
                )
            client = AsyncIsaacus()
        else:
            client = isaacus_client
        
        if isinstance(client, AsyncIsaacus):
            enrichment_result = await client.enrichments.create(
                model="kanon-2-enricher",
                texts=text,
                overflow_strategy="auto"
            )
        else:
            enrichment_result = client.enrichments.create(
                model="kanon-2-enricher",
                texts=text,
                overflow_strategy="auto"
            )

        ilgs_doc = enrichment_result.results[0].document

    # no API call, can just run directly
    return text_to_markdown(ilgs_doc, cross_references=cross_references,
                            strike_junk=strike_junk, wrap_quotes=wrap_quotes,
                            italicise_ext_refs=italicise_ext_refs,
                            isaacus_client=isaacus_client)
