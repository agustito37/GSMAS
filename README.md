# GSMAS

> Decentralized multi-agent platform for **collaborative investigation over
> knowledge graphs**, with stigmergic coordination

Given an input signal about an investigation case, specialist agents reconstruct
what happened, with what degree of confidence, and propose a conclusion — in an
auditable and reproducible way. The platform orchestrates the agents, materializes
the units of work, propagates graph events, and consolidates learning across
cases. The same architecture serves different kinds of investigation: the roles
that take part and the tools each one invokes are declared by configuration.

## Core ideas

- **Stigmergic coordination.** No agent communicates directly with another: all
  coordination happens through mutations of a shared graph. Agents react
  to graph events relevant to their role. There is no central coordinator or
  planner.
- **Distributed learning by retrospective.** When a case closes, each role refines
  its *skills* (per-role long-term memory).
- **Configurable and extensible.** Adding a new role or changing the available
  tools is declarative and does not require touching the core architecture.

## Stack

- **Python 3.14** with [`uv`](https://docs.astral.sh/uv/) for dependency management.
- **Neo4j 5** (Docker) as the knowledge graph; Neo4j Browser for inspection.
- **Direct LLM SDKs** (`openai`) behind a thin in-house interface.
- **Pydantic v2** for node models and structured LLM outputs.
- **asyncio** for the event bus and asynchronous coordination.
- **Ruff** (lint + format) and **Pyright** (type checking).

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Docker (for Neo4j)
- An OpenAI API key

## Getting started

```bash
# 1. Dependencies
uv sync

# 2. Environment variables
cp .env.example .env
#    edit .env: OPENAI_API_KEY and NEO4J_AUTH (user/password format)

# 3. Neo4j
docker compose up -d

# 4. Check
#    Neo4j Browser: http://localhost:7474
```

| Service        | Access                     |
| -------------- | -------------------------- |
| Neo4j Browser  | http://localhost:7474      |
| Neo4j Bolt     | bolt://localhost:7687      |
| User           | `neo4j` (per `NEO4J_AUTH`) |

## Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # formatting (indentation, quotes, imports)
uv run pyright           # type checking
uv run pytest            # tests (as they are added)
```
