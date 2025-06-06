# RAG repository

Repository for code related to the RAG training.

## Prerequisites

1. You will need to have `uv` installed. See how to install it [here](https://docs.astral.sh/uv/getting-started/installation/).
2. You will need an OpenAI API key. You can get one [here](https://platform.openai.com/account/api-keys). After that, you can set it in the `.env` file.
3. To run the RAG template, you will need to have `node` and `pnpm` installed. You can install them from [here](https://nodejs.org/en/download/) and [here](https://pnpm.io/installation), respectively.

## Part 1 examples

You can find the scripts related to the examples I demoed in part 1 of the training in the `part1-examples` folder.

## RAG template

You can find the RAG template I used in the training in the `rag-template` folder.

The way to use the template is to edit the `rag-template/api/index.py` file and to modify the
`STEP` variable to the step you want to run. The steps are:

- 0: A simple chatbot (no tools, no RAG)
- 1: A chatbot with web search-based RAG
- 2: A chatbot with similarity search-based RAG
- 3: Like step 2, but with an additional step to refine queries
- 4: A chatbot using a weather API (to demonstrate function calling)
- 5: A chatbot that can use multiple tools (web search, similarity search, weather API)
- 6: A research agent leveraging web search and similarity search
