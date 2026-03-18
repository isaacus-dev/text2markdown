# Text2Markdown
Efficient Python library for converting plain text into markdown format.
Text2Markdown relies on a model to parse plain text, for which the 
[Kanon 2 Enricher](https://docs.isaacus.com/capabilities/enrichment) was used.

Text2Markdown, being AI-powered, automatically infers key relationships within plain text. 

It is currently capable of:

- Segmenting text into sections with headings based on their hierarchical structure
- Embedding hyperlinks towards referenced sections
- Italicising external references 
- Detecting block quotations 
- Striking-through junk text

## Usage
### Installation 
To install text2markdown, run 
```
pip install text2markdown
```
An [Isaacus API key](https://platform.isaacus.com/accounts/signup) is required to use this library. 

### Example
Below is a short demo demonstrating the different ways you are able to use Text2Markdown.
```python
from text2markdown.t2md import text_to_markdown
from os import environ

# Add Isaacus API key as an environment variable
environ["ISAACUS_API_KEY"] = "INSERT_YOUR_API_KEY_HERE"

text = """
Section 1
I prefer markdown over plain text.
Reason 1:
I can tell what I am seeing.
Furthermore, text like in Section 2 become readable.

Section 2 
bar once famously said:
foo is a bar if and only if bar.
foo was not happy.
"""

# Run text_to_markdown on input text
output = text_to_markdown(text=text)
print(output) 
```
Alternatively, supply an [ILGS document](https://docs.isaacus.com/ilgs/introduction):
```python
from isaacus import Isaacus

# Initialise client
client = Isaacus()

# Create ILGS document
ilgs_doc = client.enrichments.create(
    model="kanon-2-enricher", 
    texts=text,
    overflow_strategy="auto"
).results[0].document

# Run text_to_markdown
output = text_to_markdown(
    text=ilgs_doc,
    isaacus_client=client
)
print(output)

```
By default, `text_to_markdown` includes all supported features.
Optional parameters can be configured to disable unwanted features:
```python
text_to_markdown(
    text=text,
    cross_references=False,
    strike_junk=False,
    wrap_quotes=False,
    italicise_ext_refs=False,
)
```
An asynchronous option is available:
```python
import asyncio
from text2markdown.async_t2md import text_to_markdown_async

async def foo():
    result = await text_to_markdown_async(text)
    print(result)

asyncio.run(foo())
```
## License
This library is licensed under the MIT License. 
