# Model Context Protocol (MCP) Server Playbook

## 🚦 Zero‑Ambiguity Quickstart (Do *exactly* this)

> Copy this section verbatim into your LLM. It contains every file and command needed for a working MCP server the first time.

### A) Create files

**`requirements.txt`**

```
mcp[cli]>=1.9.0
pydantic>=2.7.0
pytest>=8.0.0
pyyaml>=6.0.2
python-slugify>=8.0.4
```

**`src/server.py`**

```python
#!/usr/bin/env python3
# Minimal, production-safe MCP server (stdio by default; HTTP optional)
import os
from datetime import datetime, timezone
from typing import Dict, Any
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo")

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

class EchoInput(BaseModel):
    text: str = Field(description="Text to echo")
    upper: bool = Field(default=False, description="Return uppercase if true")

@mcp.tool()
async def echo(input: EchoInput) -> Dict[str, Any]:
    s = input.text.upper() if input.upper else input.text
    return {"original": input.text, "transformed": s, "timestamp": now_iso()}

@mcp.prompt()
async def server_help() -> str:
    return (
        "# Usage
"
        "List tools; call `echo` with {text, upper?}. Returns JSON with original, transformed, timestamp.
"
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Demo MCP Server")
    parser.add_argument("--transport", choices=["stdio","http","sse"], default=os.getenv("MCP_TRANSPORT","stdio"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_SERVER_PORT","8848")))
    parser.add_argument("--host", default=os.getenv("MCP_SERVER_HOST","127.0.0.1"))
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
```

**`scripts/example_client.py`** (stdio smoke test using the official SDK client)

```python
import asyncio, os
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def main():
    server = StdioServerParameters(
        command=os.environ.get("PYTHON", "python"),
        args=["src/server.py", "--transport", "stdio"],
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools, _ = await session.list_tools()
            print("TOOLS:", [getattr(t, "name", None) for t in (tools.tools if hasattr(tools, "tools") else tools)])
            result, _ = await session.call_tool("echo", {"text": "hello", "upper": True})
            content = result[0].content[0]
            print(getattr(content, "text", content))

if __name__ == "__main__":
    asyncio.run(main())
```

**`Makefile`**

```
PY ?= python3
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

$(VENV):
	$(PY) -m venv $(VENV)

bootstrap: $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

run: bootstrap
	$(PYTHON) src/server.py --transport stdio

test-stdio: bootstrap
	$(PYTHON) scripts/example_client.py

run-http: bootstrap
	$(PYTHON) src/server.py --transport http --host 127.0.0.1 --port 8848
```

### B) Install & run

```bash
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
./venv/bin/python src/server.py --transport stdio
```

### C) Verify via official client (stdio)

```bash
./venv/bin/python scripts/example_client.py
# Expect: TOOLS includes "echo" and printed JSON with transformed "HELLO"
```

### D) Wire up Claude Desktop (pick your OS)

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

Add under `mcpServers`:

```json
{
  "mcpServers": {
    "demo": {
      "command": "/ABS/PATH/TO/venv/bin/python",
      "args": ["/ABS/PATH/TO/src/server.py", "--transport", "stdio"]
    }
  }
}
```

Restart Claude Desktop → open Tools list → you should see **demo**.

### E) Optional: HTTP transport

Run server:

```bash
./venv/bin/python src/server.py --transport http --host 127.0.0.1 --port 8848
```

Then point an MCP‑HTTP‑capable client at `http://127.0.0.1:8848/` per its connector docs.

### ✅ Definition of Done (MCP server boots correctly)

* `echo` appears in the client’s **List Tools**.
* Calling `echo` returns `{ original, transformed, timestamp }`.
* No unhandled exceptions on startup or tool call.

---

*A field-tested guide for building robust MCP servers in Python (with FastMCP + the official MCP SDK), shipping via **stdio** or **HTTP**, and integrating with Claude Desktop and other MCP clients. Use this as a blueprint you can hand to an LLM to scaffold a working server end‑to‑end.*

> **How to use this playbook**
>
> * Copy sections into your LLM context when you want it to build a new server.
> * Keep the **Structure**, **Bootstrap**, **Server Skeleton**, **Transport**, and **Client Config** sections together — they are the minimum for a working server.
> * The **Patterns**, **Testing**, **Docker**, **CI**, and **Troubleshooting** sections help harden your server for real workflows.

---

## 0) Core concepts (quick refresher for the LLM)

* **MCP** is an open protocol for connecting LLM apps (clients) to **servers** that expose tools, resources, prompts, and knowledge.
* An **MCP Server**:

  * Defines **tools** (callable RPCs), optionally **prompts**, and sometimes **resources** (e.g., files/data).
  * Speaks **JSON‑RPC** over a **transport** (typically **stdio** for local, or **Streamable HTTP** for remote; some clients still support **SSE**).
* An **MCP Client** (e.g., Claude Desktop) connects to your server, lists tools/prompts, and calls them with structured arguments.

**Mental model:** treat an MCP server like a small, typed microservice that lives on your machine and the LLM can call.

---

## 1) Project structure (template)

> **Contract for LLMs:** Use *exactly* these folders and filenames. Do not invent extra layers unless asked.

```text
my-mcp-server/
├── src/
│   ├── server.py                # entrypoint with FastMCP (stdio or HTTP)
│   ├── models.py                # pydantic schemas for tools
│   ├── logic/                   # pure functions only
│   │   ├── domain.py
│   │   └── io.py
│   └── __init__.py
├── prompts/                     # optional task recipes
│   └── my_prompts.json
├── tests/
│   ├── test_direct.py           # direct tool import
│   └── test_stdio_client.py     # stdio client smoke test
├── scripts/
│   ├── example_client.py        # provided above
│   └── quickstart.sh/.ps1/.bat
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── Makefile
├── requirements.txt
└── LICENSE
```

Use this minimal, scalable layout. It mirrors what we’ve proven in production and is friendly to LLM-assisted development.

```
my-mcp-server/
├── src/
│   ├── server.py                # your entrypoint with FastMCP or the official SDK
│   ├── models.py                # pydantic models for tool inputs/outputs
│   ├── logic/                   # pure functions; keep IO thin in server.py
│   │   ├── domain.py
│   │   └── io.py
│   ├── utils/
│   │   ├── fs.py
│   │   └── ids.py
│   └── __init__.py
├── prompts/                     # optional: task recipes for LLMs
│   └── my_prompts.json
├── tests/
│   ├── test_server.py           # stdio smoke test via MCP client
│   ├── test_direct.py           # direct import/unit tests
│   └── fixtures.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml       # optional HTTP mode
├── scripts/
│   ├── quickstart.ps1/.bat/.sh  # setup helpers (venv + run)
│   ├── generate_claude_config.py
│   └── example_client.py        # official SDK client sample (stdio)
├── mcp-servers.local.json       # local Claude/Client config example
├── requirements.txt
├── Makefile
└── LICENSE
```

**Conventions that help LLMs:**

* Keep **pure logic** in `logic/` and only wire **I/O** in `server.py`.
* Put **pydantic** models for tool I/O in `models.py`.
* Provide **quickstart** scripts so the LLM can run/tests fast.

---

## 2) Python bootstrap

**Dependencies (pin major versions):**

```
mcp[cli]>=1.9.0
pydantic>=2.7.0
python-slugify>=8.0.4    # if you’ll generate-safe slugs/IDs
pyyaml>=6.0.2            # if you’ll render YAML front matter
pytest>=8.0.0            # tests
```

**Virtualenv & install (Unix):**

```bash
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

**Windows PowerShell:**

```powershell
py -3 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\pip.exe install -r requirements.txt
```

---

## 3) Server skeleton (FastMCP + official SDK): Rules for agents

> **LLM must follow:**
>
> 1. Every tool has a `BaseModel` input with `description` on every field.
> 2. Return **small, structured JSON**; never dump large blobs.
> 3. Default transport = **stdio**; accept flags for HTTP.
> 4. Provide at least one `@mcp.prompt()` with usage instructions.
> 5. Include a `main()` that switches on `--transport` and supports `--host/--port`.

*(The full minimal example lives in the Zero‑Ambiguity section and does not need to be re-generated.)*

> This template gives you: stdio by default, optional HTTP, typed tools, prompts, and a clean structure that plays well with Claude Desktop and other MCP clients.

```python
#!/usr/bin/env python3
# src/server.py

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP  # Provided by the official SDK

mcp = FastMCP("myserver")

# ---------- Utilities ----------

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

DEFAULTS: dict[str, str] = {"project_path": ""}

# ---------- Typed inputs/outputs ----------

class ExampleInput(BaseModel):
    text: str = Field(description="Some input text")
    upper: bool = Field(default=False, description="Return uppercase?")

class ExampleResult(BaseModel):
    original: str
    transformed: str
    timestamp: str

# ---------- Tools ----------

@mcp.tool()
async def echo(input: ExampleInput) -> Dict[str, Any]:
    s = input.text.upper() if input.upper else input.text
    return ExampleResult(original=input.text, transformed=s, timestamp=now_iso()).model_dump()

# ---------- Optional prompts ----------

@mcp.prompt()
async def summarize_how_to_use(text: str) -> str:
    return (
        "# How to Use This Server\n"
        "1) list tools -> 2) call `echo` with {text, upper?}\n"
        "Return JSON with original + transformed + timestamp.\n"
    )

# ---------- Entrypoint ----------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="My MCP Server")
    parser.add_argument("--transport", choices=["stdio","http","sse"], default=os.getenv("MCP_TRANSPORT","stdio"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_SERVER_PORT","8848")))
    parser.add_argument("--host", default=os.getenv("MCP_SERVER_HOST","127.0.0.1"))
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
```

**Run (stdio):**

```bash
./venv/bin/python src/server.py --transport stdio
```

**Run (HTTP):**

```bash
./venv/bin/python src/server.py --transport http --host 127.0.0.1 --port 8848
```

> Tip: keep `stdio` as default; most local clients (like Claude Desktop) assume it.

---

## 4) Transport choices & when to use them (decision table)

| Scenario                                 | Transport                  | Why                              | Server flag                                     |
| ---------------------------------------- | -------------------------- | -------------------------------- | ----------------------------------------------- |
| Local development with Claude Desktop    | **stdio**                  | Simplest, zero networking        | `--transport stdio`                             |
| Localhost web client or multiple clients | **HTTP (Streamable HTTP)** | Multi-connection, stream support | `--transport http --host 127.0.0.1 --port 8848` |
| Legacy client expecting SSE              | **SSE** (legacy)           | Compatibility only               | `--transport sse`                               |

**Rule of thumb:** prefer **stdio** for desktop/local; use **HTTP** for containers/cloud.

* **stdio** — simplest for local client ↔ server on the same machine; best DX; no networking.
* **Streamable HTTP** — the modern remote transport replacing legacy SSE: good for containerized/cloud servers; easy to reverse‑proxy; supports streaming.
* **SSE** — supported by some clients; increasingly deprecated in frameworks. Use only if your client mandates it.

**Server switches**

```
--transport stdio    # default for local
--transport http     # remote/localhost via HTTP (streamable)
--transport sse      # legacy; only if required
```

---

## 5) Defining tools the LLM can reliably call

### 5.1 Use Pydantic for argument schemas

* Create `BaseModel` classes for each tool’s input/output.
* Keep fields **explicit** and **documented** (`description=`). LLMs learn from these.
* Validate and normalize early; return **clear error messages**.

**Pattern:** a small, pure function per tool that delegates to domain logic.

```python
class CreateItemInput(BaseModel):
    name: str = Field(description="Human-readable name")
    priority: int = Field(ge=1, le=5, description="1 (low) … 5 (high)")

@mcp.tool()
async def create_item(input: CreateItemInput) -> dict:
    # domain logic
    item = {"id": f"IT-{int(datetime.now().timestamp())}", **input.model_dump()}
    return {"status": "ok", "item": item}
```

### 5.2 Prompts for agent guidance

Prompts act like built‑in manuals or task recipes; clients can list and read them.

```python
@mcp.prompt()
async def commit_help() -> str:
    return (
        "# Commit Format\n"
        "Subject: `[pm <ID>] <type>: <summary>`\n"
        "Trailer: `PM: <ID>`\n"
    )
```

### 5.3 Resources (optional)

If your server must expose a file tree or data blobs, implement **resources** APIs (check your SDK version’s support). Keep resource reads **bounded** (size/line limits) and consider **allowlists** for paths.

---

## 6) Client integration (Claude Desktop & others) — exact steps

1. **Locate config**

   * macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   * Windows: `%APPDATA%/Claude/claude_desktop_config.json`
2. **Add server block** under `mcpServers` with **absolute** paths.
3. **Restart** the app, open the tools list, confirm your server appears.

**Template**

```json
{
  "mcpServers": {
    "demo": {
      "command": "/abs/path/to/venv/bin/python",
      "args": ["/abs/path/to/src/server.py", "--transport", "stdio"]
    }
  }
}
```

**CLI alternative**

```bash
claude mcp add demo -- "$(pwd)/venv/bin/python" "$(pwd)/src/server.py" --transport stdio
```

### 6.1 Fast path: Claude CLI add

```bash
# inside your project
claude mcp add myserver -- "$(pwd)/venv/bin/python" "$(pwd)/src/server.py" --transport stdio
```

### 6.2 Manual config file

Mac/Linux (commonly): `~/.claude.json`
Windows: `%USERPROFILE%\AppData\Roaming\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "myserver": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": ["/absolute/path/to/src/server.py", "--transport", "stdio"]
    }
  }
}
```

> After editing, restart the client and look for your server in the MCP tool list.

---

## 7) Testing your server (no-surprises recipe)

### Direct unit test

```python
# tests/test_direct.py
import asyncio
from src.server import echo, EchoInput

def run(coro):
    return asyncio.run(coro)

def test_echo_upper():
    out = run(echo(EchoInput(text="hi", upper=True)))
    assert out["transformed"] == "HI"
```

### Stdio smoke test (client over MCP)

```python
# tests/test_stdio_client.py
import asyncio, os
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def test_stdio_smoke():
    server = StdioServerParameters(command=os.environ.get("PYTHON","python"), args=["src/server.py","--transport","stdio"]) 
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools, _ = await session.list_tools()
            names = [getattr(t, 'name', None) for t in (tools.tools if hasattr(tools,'tools') else tools)]
            assert 'echo' in names
```

**Run**

```bash
pytest -q
```

### 7.1 Direct tests (imports only)

* Unit test your domain logic with **pytest**.
* Optionally import your tool coroutine and run it with `asyncio.run()`.

```python
# tests/test_direct.py
import asyncio
from src.server import echo, ExampleInput

def run(coro):
    return asyncio.run(coro)

def test_echo_upper():
    out = run(echo(ExampleInput(text="hi", upper=True)))
    assert out["transformed"] == "HI"
```

### 7.2 Smoke test via stdio client

Use the official Python SDK client to start your server as a subprocess and call tools over MCP (like a real client would).

```python
# scripts/example_client.py
import asyncio, os
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def main():
    server = StdioServerParameters(
        command=os.environ.get("PYTHON", "python"),
        args=["src/server.py", "--transport", "stdio"],
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result, _ = await session.call_tool("echo", {"text": "hello", "upper": True})
            content = result[0].content[0]
            print(getattr(content, "text", content))

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 8) Production‑readiness patterns

### 8.1 ID generation & safe paths

* Generate **deterministic IDs** per period (e.g., `KEY-YYYYMM-###`).
* Always ensure file operations remain **under a project root** (reject path traversal).

```python
from pathlib import Path

def ensure_under(base: Path, target: Path) -> Path:
    base = base.resolve(); target = target.resolve()
    target.relative_to(base)  # raises if escape
    return target
```

### 8.2 Conventional commit helper (example)

Map domain types to conventional commit types and return **branch/commit hints** from tools.

```python
TYPE_TO_CC = {"feature": "feat", "bug": "fix", "refactor": "refactor", "chore": "chore", "spike": "spike"}

def cc_type(t: str) -> str:
    return TYPE_TO_CC.get((t or "").lower(), "feat")
```

### 8.3 Atomic workflow tool

Bundle several actions (edit, validate, set status, optional commit) into a single **idempotent** tool to reduce agent error.

```python
class AtomicUpdateInput(BaseModel):
    node_id: str
    summary: Optional[str] = None
    plan: Optional[str] = None
    acceptance: Optional[List[str]] = None
    status: Optional[str] = None
    check_acceptance_all_done: bool = False
    commit: bool = False
    commit_message: Optional[str] = None
```

### 8.4 Logging & observability

* Prefer **structured logs** (JSON) on the server boundary.
* Log **tool name**, **args schema version**, **duration**, **success/error**.
* For HTTP, put your server behind a reverse proxy with **request IDs**.

---

## 9) Docker & Compose (optional)

**Dockerfile (slim + non‑root):**

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash mcp && mkdir -p /app /workspace && chown -R mcp:mcp /app /workspace
USER mcp
WORKDIR /app
COPY --chown=mcp:mcp requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=mcp:mcp src/ ./src/
ENV PYTHONUNBUFFERED=1 PATH=/home/mcp/.local/bin:$PATH MCP_WORKSPACE_DIR=/workspace
VOLUME ["/workspace"]
CMD ["python", "src/server.py", "--transport", "stdio"]
```

**Compose for HTTP transport:**

```yaml
version: '3.8'
services:
  myserver:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: my-mcp
    command: ["python", "src/server.py", "--transport", "http", "--port", "8848"]
    ports: ["8848:8848"]
    volumes: ["${WORKSPACE_DIR:-./workspace}:/workspace"]
    environment: ["MCP_WORKSPACE_DIR=/workspace"]
```

---

## 10) Makefile targets (DX accelerators)

```makefile
PY ?= python3
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
SERVER := src/server.py

$(VENV):
	$(PY) -m venv $(VENV)

bootstrap: $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

run: bootstrap
	$(PYTHON) $(SERVER) --transport stdio

claude-add: bootstrap
	claude mcp add myserver -- "$(abspath $(PYTHON))" "$(abspath $(SERVER))" --transport stdio

docker-build:
	docker build -f docker/Dockerfile -t my-mcp:latest .
```

---

## 11) Security & safety checklist (musts for LLMs)

* **No path traversal**: implement and use `ensure_under(base, target)` for *all* file writes/reads.

* **Strict schemas**: all tool inputs are `BaseModel`; reject extras (use `model_config = {'extra': 'forbid'}` if needed).

* **Rate/size limits**: cap lines and bytes for reads; paginate results.

* **Side-effects**: destructive ops require explicit booleans like `confirm=True`.

* **HTTP auth**: if exposing over HTTP, place behind a reverse proxy and require auth headers or session tokens.

* **Secrets**: never echo secrets; mask in logs; read from env/files, not arguments.

* **Path safety:** resolve & constrain all file operations under an allowlisted base directory.

* **Argument validation:** strict `pydantic` models; reject unknown fields via `model_config`.

* **Resource limits:** cap file sizes, line counts, recursion depths; paginate results.

* **Side‑effects policy:** make destructive tools **opt‑in** and require explicit confirmations (flags).

* **AuthN/Z (HTTP):** use reverse proxy or middleware for auth if you expose the server remotely.

* **Secrets:** never echo secrets back; support file‑based or env‑based credentials; avoid logging secrets.

---

## 12) Troubleshooting playbook (checklist order)

1. **Server doesn’t show up in client** → Confirm absolute paths in config; run server manually; watch for import errors.
2. **No tools listed** → Ensure `@mcp.tool()` decorators are present and imported; verify `await session.initialize()` is called in client.
3. **HTTP not reachable** → Confirm `--host/--port`; verify firewall; if behind proxy, allow chunked/SSE.
4. **Tool schema misused by agent** → Add field `description`s; return example payloads; keep outputs short and consistent.
5. **Windows path issues** → Use absolute paths and escape backslashes in JSON; prefer PowerShell for bootstraps.

**Client can’t see my server**

* Verify the config path and exact command/args.
* Test the binary paths are absolute; clients often require absolute paths.
* Run your server manually; check it prints no errors on startup.

**Tools list is empty**

* Ensure your functions are decorated with `@mcp.tool()` / `@mcp.prompt()` and imported in `server.py`.

**HTTP transport not working**

* Confirm port is open and not blocked by firewall.
* If behind a proxy, verify request/response bodies are passed through unmodified and chunked encoding is supported.

**LLM keeps calling tools with wrong shapes**

* Tighten pydantic schemas, add `description=` on every field, and return example payloads in success responses.

**Git operations fail in ephemeral temp dirs**

* Ensure you initialize `git` and set a test user/email in tests.

---

## 13) Copy‑paste scaffolds

### 13.1 `requirements.txt`

```
mcp[cli]>=1.9.0
pydantic>=2.7.0
pytest>=8.0.0
pyyaml>=6.0.2
python-slugify>=8.0.4
```

### 13.2 `scripts/quickstart.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
./venv/bin/python src/server.py --transport stdio
```

### 13.3 Claude config generator (optional)

```python
# scripts/generate_claude_config.py
import json, platform, subprocess
from pathlib import Path

base = Path(__file__).parent.parent.resolve()
python_path = base / ("venv/Scripts/python.exe" if platform.system()=="Windows" else "venv/bin/python")
server_path = base / "src" / "server.py"

cfg = {"mcpServers": {"myserver": {"command": str(python_path), "args": [str(server_path), "--transport", "stdio"]}}}
print(json.dumps(cfg, indent=2))
```

---

## 14) Design patterns that work well with agents

* **Idempotent tools**: Safe to retry (e.g., compute, validate, list).
* **Atomic update tools**: Perform multi‑step workflows with internal validation; return a summary of changes.
* **Hints as outputs**: Return `branch_hint`, `commit_preamble`, `commit_trailer`, or any guidance the agent can reuse in later steps.
* **Small responses**: Tools should return compact, structured JSON; large blobs should be resources or separate read APIs with pagination/limits.

---

## 15) LLM usage instructions (paste alongside this playbook)

When you, the LLM, generate a server:

1. Use the **Project structure** and **Server skeleton** above.
2. Default to **stdio** unless explicitly told to use HTTP.
3. Every tool MUST define: a pydantic **input** model and clear **output** fields.
4. Add at least one **prompt** named `*_help` that explains the workflow.
5. Output clear **run commands** and a **Claude config snippet**.
6. Provide a minimal **pytest** that calls one tool and asserts a field.
7. If the server touches files, implement `ensure_under(base, target)` and enforce it.
8. Avoid printing raw secrets or large responses; paginate.

---

## 16) Appendix — Reference snippets

**Front‑matter template replacement**

```python
def render_template(t: str, values: dict[str, str]) -> str:
    for k, v in values.items():
        t = t.replace(f"{{{{{k}}}}}", v)
    return t
```

**Markdown section replace/extract**

```python
def extract_section(body: str, header: str) -> str:
    lines, grab, buf = body.splitlines(), False, []
    for line in lines:
        if line.strip().lower() == f"# {header}".lower():
            grab = True; continue
        if grab and line.startswith("# "): break
        if grab: buf.append(line)
    return "\n".join(buf).strip()
```

**Git status (porcelain) parsing**

```python
import subprocess

def run_git(args: list[str], cwd: Path) -> dict:
    try:
        res = subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True, text=True, check=True)
        return {"success": True, "output": (res.stdout or "").strip()}
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": (e.stderr or e.stdout or str(e)).strip()}
```

---

## 17) Protocol essentials (no‑ambiguity, implementation targets)

### 17.1 JSON‑RPC envelope (over stdio or HTTP)

* **Encoding:** UTF‑8 JSON. Each message is a complete JSON object.
* **Base fields:** `jsonrpc:"2.0"`, `id` (for requests/responses), `method`, `params`.
* **Result shape:** `{ "jsonrpc":"2.0", "id": <same>, "result": { ... } }`.
* **Error shape:** `{ "jsonrpc":"2.0", "id": <same>, "error": { "code": <int>, "message": <str>, "data": { ...optional } } }`.
* **Do not** stream partial JSON on stdio; send full JSON per line or frame.

### 17.2 Required lifecycle calls (client ↔ server)

1. **initialize** → server returns capabilities (what you implement: tools, prompts, resources, transport features).
2. **tools/list** → enumerate tool metadata (name, description, JSON schema for params).
3. **prompts/list** (optional) → enumerate prompt names + descriptions.
4. **tool/call** → execute tool with validated params, return structured result content.
5. **notifications** (optional) → e.g., `tools/list_changed` when dynamic toolset changes.

### 17.3 Tool descriptors (what you must provide)

* **Unique name**: kebab or snake case, stable.
* **Description**: imperative, single line.
* **JSON Schema** for params: generated from Pydantic; ensure every field has `description` and types are concrete (no `Any`).
* **Examples** (optional but recommended): brief sample `params` in the description.

### 17.4 Result content types (what you can return)

* **Text content**: `{ type: "text", text: "..." }` for human‑readable strings.
* **Resource links**: `{ type: "resource_link", uri, name, mimeType?, size? }` for files/data the client can fetch via resources APIs.
* **Structured JSON**: return as normal fields inside `result`; keep it compact and documented.
* Prefer **links over blobs** for anything large; paginate long text.

### 17.5 Streaming semantics (HTTP transport)

* Client sends **HTTP POST** with JSON‑RPC body.
* For streaming responses, server upgrades a **response channel via SSE**; client MUST send `Accept: text/event-stream` to open it.
* Send final JSON‑RPC `result` (or `error`) event to close the call; do not intermingle different `id`s on one stream.

### 17.6 Error codes (recommended subset)

* `-32601` **Method not found** (unknown tool or RPC).
* `-32602` **Invalid params** (pydantic validation fails; include field errors in `error.data`).
* `-32000` **Server error** (unexpected); include minimal diagnostics in `error.data.code` and a short `hint`.
* For domain errors, use `-32001..-32099` with documented meanings.

### 17.7 Concurrency, timeouts, backpressure

* Mark tools **async** and keep them non‑blocking; run blocking IO in a thread pool.
* Enforce **per‑tool timeouts**; return a precise timeout error.
* Limit in‑flight calls (e.g., semaphore) and reject beyond capacity with a clear error.

### 17.8 Versioning and compatibility

* Pin your SDK/framework versions (see `requirements.txt`).
* Expose a `server_version` string in `initialize.result.capabilities`.
* Document supported transports (`stdio`, `http`, optional `sse`).

---

## 18) HTTP deployment hardening (copy/paste)

**Reverse proxy (concepts)**

* Terminate TLS at proxy; forward to server on localhost.
* Pass through `Content-Type: application/json` and `Accept: text/event-stream` for streaming.
* Set `keepalive` and `proxy_read_timeout` > tool max runtime.

**Auth patterns**

* Prefer **Bearer tokens** or API keys in `Authorization` header.
* For multi‑tenant, include `X-Org`/`X-User` headers and validate.
* Never log auth headers.

**CORS** (only if the client is browser‑based): set exact origins and methods; disallow `*` in production.

---

## 19) Contract tests (ensure the protocol is honored)

Create **deterministic** tests that exercise the protocol, not just Python functions.

```python
# tests/test_contract_stdio.py
import asyncio, os
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def test_contract():
    server = StdioServerParameters(command=os.environ.get("PYTHON","python"), args=["src/server.py","--transport","stdio"]) 
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools, _ = await session.list_tools()
            names = [getattr(t, 'name', None) for t in (tools.tools if hasattr(tools,'tools') else tools)]
            assert 'echo' in names
            # Invalid params → JSON‑RPC error
            result, _ = await session.call_tool("echo", {"text": 123})
            # Expect an error object in response; assert client raises or returns error
```

Add a second test for **HTTP** transport behind your reverse proxy.

---

## 20) JSON‑RPC frame examples (LLM copy‑ready)

**List tools (request)**

```json
{ "jsonrpc":"2.0", "id": 1, "method": "tools/list", "params": {} }
```

**List tools (response excerpt)**

```json
{ "jsonrpc":"2.0", "id": 1, "result": { "tools": [
  {"name":"echo","description":"Echo text","inputSchema":{"type":"object","properties":{"text":{"type":"string"},"upper":{"type":"boolean"}},"required":["text"]}}
]}}
```

**Call tool (request)**

```json
{ "jsonrpc":"2.0", "id": 2, "method": "tool/call", "params": {"name":"echo","arguments":{"text":"hello","upper":true}}}
```

**Call tool (response)**

```json
{ "jsonrpc":"2.0", "id": 2, "result": { "content": [ {"type":"text","text":"{\"original\":\"hello\",\"transformed\":\"HELLO\"}"} ] } }
```

> Many clients also accept structured objects in `result` — keep shapes compact and documented.

---

## 21) Resource links & safe file serving

* When returning files, prefer **resource links** with a `uri` that your server can resolve via its `resources/read` API.
* Include `mimeType`, `name`, `size` when known.
* Implement an allowlist of readable roots; reject anything outside allowlist.

---

## 22) Windows/WSL specifics (edge cases that break agents)

* Use **absolute paths** with escaped backslashes in JSON.
* When using WSL with a desktop client, ensure the `command` path resolves inside the same environment the client can launch.
* Prefer PowerShell for bootstrap scripts and quote arguments containing backslashes.

---

### End

*This playbook is intentionally verbose and structured so agents can copy sections verbatim to scaffold a working MCP server.*

