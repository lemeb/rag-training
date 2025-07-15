# RAG Template

## Launching the app

To run the example locally you need to:

1. Ensure you have stored your `OPENAI_API_KEY` in the `.env` file in the root directory of the project. (Not in this folder, but the parent of this folder.)
2. `pnpm install` to install the required Node dependencies.
3. `uv venv` to create a virtual environment.
4. `source venv/bin/activate` to activate the virtual environment.
5. `uv sync` to install the required Python dependencies.
6. `pnpm dev` to launch the development server.
7. Open your browser and navigate to `http://localhost:3000` to see the app in action.

## Switching between steps

The way to use the template is to edit the `api/index.py` file and to modify the
`STEP` variable to the step you want to run. The steps are:

- 0: A simple chatbot (no tools, no RAG)
- 1: A chatbot with web search-based RAG
- 2: A chatbot with similarity search-based RAG
- 3: Like step 2, but with an additional step to refine queries
- 4: A chatbot using a weather API (to demonstrate function calling)
- 5: A chatbot that can use multiple tools (web search, similarity search, weather API)
- 6: A research agent leveraging web search and similarity search

### If DuckDuckGo is not working

If DuckDuckGo does not perform search (because of rate limiting), you can
use [Exa](https://exa.ai/) as an alternative. To do so, you need to sign up for an account and then add your API key as `EXA_API_KEY` to the `.env` file in the root directory of the project.

## Don't forget to run the RAG pre-processing!

Note that the RAG pre-processing will _not_ have been done yet! To do so, you should run the  following command:

```bash
# Run this from the 'rag-template' directory
uv run python -m api.utils.pdf
```


