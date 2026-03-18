""" Async version of t2md. """
from functools import partial
from typing import TYPE_CHECKING
import os
import asyncio

from text2markdown.t2md import text_to_markdown

from isaacus import AsyncIsaacus

if TYPE_CHECKING:
    import isaacus
    from isaacus.types.ilgs.v1.document import Document as ILGSDocument

async def text_to_markdown_async(
    text: "str | ILGSDocument",
    *,
    cross_references: bool = True,
    strike_junk: bool = True,
    wrap_quotes: bool = True,
    italicise_ext_refs: bool = True,
    isaacus_client: "isaacus.Isaacus | None" = None,
) -> str:

    if isinstance(text, str):
        api_key = os.getenv("ISAACUS_API_KEY")
        if api_key is None:
            raise ValueError(
                """ Could not find an Isaacus API key in environment variables. See https://platform.isaacus.com/accounts/signup/."""
            )

        # need to make API call, run in thread pool
        client = AsyncIsaacus()
        enrichment_result = await client.enrichments.create(
            model="kanon-2-enricher",
            texts=text,
            overflow_strategy="auto"
        )

        ilgs_doc = enrichment_result.results[0].document
        loop = asyncio.get_running_loop()
        text_to_ilgs = partial(
            text_to_markdown,
            ilgs_doc,
            cross_references=cross_references,
            strike_junk=strike_junk,
            wrap_quotes=wrap_quotes,
            italicise_ext_refs=italicise_ext_refs,
            isaacus_client=client,
        )
        return await loop.run_in_executor(None, text_to_ilgs)

    # no API call, can just run directly
    return text_to_markdown(text, cross_references=cross_references,
                            strike_junk=strike_junk, wrap_quotes=wrap_quotes,
                            italicise_ext_refs=italicise_ext_refs,
                            isaacus_client=isaacus_client)