from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import asyncio

from dotenv import load_dotenv

from text2markdown.t2md import text_to_markdown as t2md
from text2markdown.async_t2md import text_to_markdown_async as t2mda

if TYPE_CHECKING:
    from os import PathLike

TEST_DIR = Path(__file__).parent.resolve() / "test-in"
TEST_OUT_DIR = Path(__file__).parent.resolve() / "test-out"


def get_text(path: PathLike) -> str:
    """ Read from `path`. """
    text = ""
    with open(Path(path), "r") as f:
        for line in f.readlines():
            text += line
    return text


def no_api_key() -> bool:
    """ Call `text_to_markdown` with no API key. """
    tests = [test for test in TEST_DIR.iterdir() if test.is_file()]
    text = get_text(tests[0])

    try: 
        t2md(text)
        err = False
    except ValueError: 
        err = True

    return err


def run_default() -> None:
    """ Calls `text_to_markdown` on all tests in `/test_in`
    with default args and saves output in `/test_out`
    """
    tests = sorted([test for test in  TEST_DIR.iterdir() if test.is_file() and test.suffix == ".txt"])
    for test in tests:
        txt = t2md(get_text(test))
        # save test results
        with open(TEST_OUT_DIR / f"{test.stem}.md", "w") as f:
            f.write(txt)


async def run_async() -> None: 
    tests = sorted([test for test in  TEST_DIR.iterdir() if test.is_file() and test.suffix == ".txt"])
    for test in tests:
        # don't actually do anything, just check that there's no errors
        resp = await t2mda(get_text(test))
        assert(isinstance(resp, str))


async def test():
    TEST_OUT_DIR.mkdir(exist_ok=True)
    TEST_DIR.mkdir(exist_ok=True)
    assert no_api_key()

    load_dotenv()
    run_default()
    await run_async()

if __name__ == "__main__":
    asyncio.run(test())
