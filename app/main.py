import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from core.events.bus import Event
from core.graph.models import NodeBase
from core.providers.openai_provider import OpenAIProvider
from core.runtime.orchestrator import Orchestrator
from core.tools.base import ToolRegistry
from domain.roles.investigator import Investigator
from domain.roles.planner import Planner
from domain.roles.synthesizer import Synthesizer
from domain.roles.theorist import Theorist
from domain.tools.log_query import LogQueryTool

logger = logging.getLogger("haive.dashboard")

NEO4J_URI = "bolt://localhost:7687"
MODEL = "gpt-4o-mini"
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


class EventStream:
    """Bridges the bus to the connected WebSockets, each scoped to ONE workspace. Just one
    more observer of the medium: it subscribes like any role would, mutates nothing. Every
    mutation is resolved to a workspace and only forwarded to the clients viewing it."""

    def __init__(self) -> None:
        self.clients: dict[WebSocket, str] = {}  # ws -> the workspace it is viewing
        self.orchestrator: Orchestrator | None = None

    def wire(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator
        for event_type in ("node_created", "node_updated", "edge_created"):
            orchestrator.bus.subscribe(event_type, self._forward)

    async def send_snapshot(self, ws: WebSocket, workspace: str) -> None:
        if self.orchestrator is None:
            return
        snapshot = await self.orchestrator.store.get_workspace_graph(workspace)
        await ws.send_json({"type": "snapshot", "workspace": workspace, **snapshot})

    async def _forward(self, event: Event) -> None:
        payload: dict = {
            "type": event.type,
            "node_id": event.node_id,
            "node_type": event.node_type,
            **event.payload,
        }
        # resolve the node once: enrich the payload with its props (the event is only a
        # pointer) AND find its workspace, so the event reaches only the matching clients
        node_id = event.node_id or event.payload.get("from_id")
        workspace: str | None = None
        if node_id and self.orchestrator:
            node = await self.orchestrator.store.get_node(node_id)
            if node is not None:
                if event.type != "edge_created":
                    payload["props"] = node.model_dump(mode="json")
                workspace = await self._node_workspace(node)
        for ws, ws_workspace in list(self.clients.items()):
            if workspace is not None and workspace != ws_workspace:
                continue
            try:
                await ws.send_json(payload)
            except Exception:
                self.clients.pop(ws, None)

    async def _node_workspace(self, node: NodeBase) -> str:
        """The workspace a node belongs to: directly (InputSignal/Case/Role), via its
        role_id prefix (Skill/LTM: '{workspace}:{name}[...]'), or via its case (the
        case-scoped nodes)."""
        workspace = getattr(node, "workspace_id", None)
        if workspace:
            return workspace
        role_id = getattr(node, "role_id", None)
        if role_id and ":" in role_id:
            return role_id.split(":", 1)[0]
        case_id = getattr(node, "case_id", None)
        if case_id and self.orchestrator:
            cases = await self.orchestrator.store.query_nodes("Case", {"case_id": case_id})
            if cases:
                return getattr(cases[0], "workspace_id", "default")
        return "default"


stream = EventStream()


@asynccontextmanager
async def lifespan(app: FastAPI):
    user, password = os.environ["NEO4J_AUTH"].split("/", 1)
    orchestrator = Orchestrator(
        NEO4J_URI,
        user,
        password,
        tools=ToolRegistry([LogQueryTool("data/telemetry.jsonl")]),
    )
    provider = OpenAIProvider(model=MODEL, api_key=config.OPENAI_API_KEY)
    orchestrator.register(Investigator(orchestrator.store), provider=provider)
    orchestrator.register(Theorist(orchestrator.store), provider=provider)
    orchestrator.register(Planner(orchestrator.store), provider=provider)
    orchestrator.register(Synthesizer(orchestrator.store), provider=provider)
    stream.wire(orchestrator)
    await orchestrator.start()
    app.state.orchestrator = orchestrator
    yield
    await orchestrator.aclose()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def no_store(request, call_next):
    # dev dashboard with reload=True: never cache, so a refresh always serves fresh
    # HTML/JS (avoids stale-module confusion). Does not touch the WebSocket.
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/workspaces")
async def list_workspaces() -> dict:
    """The known workspaces (their Workspace nodes), plus 'default' always."""
    nodes = await app.state.orchestrator.store.query_nodes("Workspace", {})
    ids = sorted({n.id for n in nodes} | {"default"})
    return {"workspaces": ids}


@app.delete("/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str) -> dict:
    """Delete a workspace and all its nodes (the dashboard's control confirms first)."""
    deleted = await app.state.orchestrator.store.delete_workspace(workspace_id)
    logger.info("workspace %s deleted (%d nodes)", workspace_id, deleted)
    return {"deleted": deleted}


@app.post("/signal")
async def submit_signal(body: dict) -> dict:
    workspace = body.get("workspace") or "default"
    signal_id = await app.state.orchestrator.submit_signal(body["content"], workspace)
    logger.info(
        "signal submitted id=%s ws=%s content=%r", signal_id[:8], workspace, body["content"][:80]
    )
    return {"id": signal_id}


@app.post("/verdict/{verdict_id}/feedback")
async def verdict_feedback(verdict_id: str, body: dict) -> dict:
    """Human feedback on a Verdict: the gate of RQ3. Writes verdict.feedback, whose
    node_updated/Verdict mutation triggers every LearningRole's retrospective (there is no
    domain event; the graph state IS the trigger)."""
    feedback = body.get("feedback")
    if feedback not in ("correct", "incorrect", "partial"):
        raise HTTPException(status_code=400, detail="feedback must be correct, incorrect or partial")
    await app.state.orchestrator.store.update_node(verdict_id, {"feedback": feedback})
    logger.info("verdict %s feedback=%s -> retrospectives triggered", verdict_id[:8], feedback)
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    stream.clients[ws] = "default"
    await stream.send_snapshot(ws, "default")
    try:
        while True:
            # the client sends {"workspace": id} when it switches; re-scope + re-snapshot
            msg = await ws.receive_json()
            workspace = msg.get("workspace") or "default"
            stream.clients[ws] = workspace
            await stream.send_snapshot(ws, workspace)
    except WebSocketDisconnect:
        stream.clients.pop(ws, None)


# static UI (the dashboard lives in app/web/; mounted last so it does not shadow routes)
app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


if __name__ == "__main__":
    # pipeline logs (roles, claims, failures) share the console with uvicorn's
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # reload=True: picks up code changes automatically (no more stale-page confusion)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
