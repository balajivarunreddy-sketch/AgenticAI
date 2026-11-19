# Agentic AI Flow

A standalone, Python-based multi-agent workflow for planning a request, gathering
evidence with tools, producing a deliverable, and checking the result. It is not a
RAG application and does not index documents or use a vector database.

## Workflow

```text
User request
  -> Planner (validated plan)
  -> Researcher (bounded web, calculator, and workspace-file tool calls)
  -> Executor (drafts the requested work product)
  -> Evaluator (scores evidence, completeness, and clarity)
       -> Researcher and Executor for a bounded revision, if needed
       -> Final response
```

LangGraph `StateGraph` carries workflow state and conversation messages. Its SQLite
checkpointer persists conversation memory by thread ID. The default thread ID is
`default`; choose a new ID with `--thread-id` to start a separate conversation.

## Requirements

- Python 3.10 or newer
- An API key for the selected LLM provider; NVIDIA API credentials are also needed
  for the default hosted guardrails
- Internet access for the DuckDuckGo web-search tool

## Install

From this directory, in PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` if needed and set the provider credentials. The project
includes a blank `.env` template, ignored by Git; do not commit real credentials.
For NVIDIA, set `LLM_PROVIDER=nvidia` and provide `NVIDIA_API_KEY` (or `API_KEY`);
the default model is `nvidia/nemotron-3.5-lightning-30b-a3b`. For Azure OpenAI, set
`LLM_PROVIDER=azure` and fill in the Azure variables in `.env`. `OPENAI_MODEL`
is used by the OpenAI provider.

By default, guardrails send each request and final answer to NVIDIA's
`nvidia/llama-3.1-nemotron-safety-guard-8b-v3` model for safety classification.
The app blocks unsafe requests, withholds unsafe answers, and fails closed if the
safety service cannot provide a valid result. It also rejects oversized or
credential-bearing requests, redacts common credentials from retrieved file/web
content and final answers, and labels retrieved content as untrusted. This sends
user requests and final answers to NVIDIA; set `GUARDRAILS_ENABLED=false` only if
you intentionally want to disable hosted checks.

## Run

```powershell
python main.py
```

## Run with Docker

Install Docker Desktop, copy `.env.example` to `.env` if needed, and configure the
provider and NVIDIA safety credentials in `.env`. The Compose service passes those
values to the container at runtime; `.env` is excluded from the image and Git.

```powershell
docker compose build
docker compose run --rm app
```

The Compose setup mounts `workspace_data` and `data` from the project directory so
workspace files and SQLite checkpoints persist across container runs. Use `exit` to
stop the interactive app.

Example requests:

- `Research the trade-offs between SQLite and PostgreSQL for a small app.`
- `Calculate the monthly payment on a 25000 loan at 6.5% APR over 48 months.`
- `Summarize notes.txt in my workspace files.` (place `notes.txt` in `workspace_data`)

Use `exit` to leave the CLI. The web-search tool sends its search query to
DuckDuckGo. File tools are read-only, scoped to `WORKSPACE_ROOT` (defaults to
`workspace_data`), block `.env`, `.git`, and virtual-environment paths, and limit
reads to 100 KB. The executor does not perform external side effects.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | `openai`, `azure`, or `nvidia` |
| `OPENAI_API_KEY` | — | Required for OpenAI |
| `OPENAI_MODEL` | `gpt-4.1-mini` | OpenAI model name |
| `NVIDIA_API_KEY` / `API_KEY` | — | Required for NVIDIA |
| `NVIDIA_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | NVIDIA model name |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA OpenAI-compatible endpoint |
| `GUARDRAILS_ENABLED` | `true` | Enable hosted input/output safety checks |
| `NVIDIA_SAFETY_API_KEY` | `NVIDIA_API_KEY` / `API_KEY` | Optional separate safety-model key |
| `NVIDIA_SAFETY_MODEL` | `nvidia/llama-3.1-nemotron-safety-guard-8b-v3` | Safety classification model |
| `NVIDIA_SAFETY_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA safety-model endpoint |
| `MAX_INPUT_CHARS` | `10000` | Maximum user request size (1–100000) |
| `AZURE_OPENAI_API_KEY` | — | Required for Azure OpenAI |
| `AZURE_OPENAI_ENDPOINT` | — | Azure resource endpoint |
| `AZURE_OPENAI_API_VERSION` | `2024-10-21` | Azure API version |
| `AZURE_OPENAI_DEPLOYMENT` | — | Azure model deployment name |
| `WORKSPACE_ROOT` | `workspace_data` | Root exposed to read-only file tools |
| `CHECKPOINT_DB_PATH` | `data/checkpoints.sqlite` | Persistent LangGraph state |
| `LANGGRAPH_THREAD_ID` | `default` | Conversation memory key |
| `MAX_TOOL_ROUNDS` | `4` | Maximum researcher tool-call rounds (1–8) |
| `MAX_REVISIONS` | `1` | Maximum evaluator-requested retries (1–3) |
| `LOG_LEVEL` | `INFO` | Standard logging level |

Environment variables take precedence over values in `.env`.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover calculator expression restrictions and workspace file-path
boundaries; they do not call the LLM or the public web-search service.

## Project layout

```text
agents/       Planner, researcher, executor, and evaluator
config/       Environment settings, provider setup, guardrails, structured logging
graph/        LangGraph state machine and evaluator retry edge
prompts/      Per-agent system instructions
tools/        Web search, safe arithmetic, and confined read-only file access
tests/        Focused offline unit tests
main.py       Interactive CLI and SQLite checkpointer lifecycle
Dockerfile    Container image definition
compose.yaml  Interactive Compose service and persistent data mounts
```
