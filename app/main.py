import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import config
from core.events.bus import Event
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


class EventStream:
    """Bridges the bus to the connected WebSockets. Just one more observer of the
    medium: it subscribes like any role would, mutates nothing."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.orchestrator: Orchestrator | None = None

    def wire(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator
        for event_type in ("node_created", "node_updated", "edge_created"):
            orchestrator.bus.subscribe(event_type, self._forward)

    async def _forward(self, event: Event) -> None:
        payload: dict = {
            "type": event.type,
            "node_id": event.node_id,
            "node_type": event.node_type,
            **event.payload,
        }
        # enrich node events with the node's current properties (the event is only
        # a pointer; the client needs the content to draw labels/status)
        if event.node_id and event.type != "edge_created" and self.orchestrator:
            node = await self.orchestrator.store.get_node(event.node_id)
            if node is not None:
                payload["props"] = node.model_dump(mode="json")
        for ws in list(self.clients):
            try:
                await ws.send_json(payload)
            except Exception:
                self.clients.discard(ws)


stream = EventStream()


@asynccontextmanager
async def lifespan(app: FastAPI):
    user, password = os.environ["NEO4J_AUTH"].split("/", 1)
    orchestrator = Orchestrator(NEO4J_URI, user, password)
    provider = OpenAIProvider(model=MODEL, api_key=config.OPENAI_API_KEY)
    tools = ToolRegistry([LogQueryTool("data/telemetry.jsonl")])
    orchestrator.register(Investigator(orchestrator.store, provider, tools))
    orchestrator.register(Theorist(orchestrator.store, provider))
    orchestrator.register(Planner(orchestrator.store))
    orchestrator.register(Synthesizer(orchestrator.store))
    stream.wire(orchestrator)
    await orchestrator.start()
    app.state.orchestrator = orchestrator
    yield
    await orchestrator.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


@app.post("/signal")
async def submit_signal(body: dict) -> dict:
    signal_id = await app.state.orchestrator.submit_signal(body["content"])
    logger.info("signal submitted id=%s content=%r", signal_id[:8], body["content"][:80])
    return {"id": signal_id}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    stream.clients.add(ws)
    snapshot = await app.state.orchestrator.store.get_full_graph()
    await ws.send_json({"type": "snapshot", **snapshot})
    try:
        while True:
            await ws.receive_text()  # the client sends nothing; this parks until disconnect
    except WebSocketDisconnect:
        stream.clients.discard(ws)


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>haive - live graph</title>
<script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
<style>
  body { margin: 0; font-family: ui-monospace, SFMono-Regular, monospace; display: flex; height: 100vh; }
  #cy { flex: 1; }
  #side { width: 360px; border-left: 1px solid #ddd; padding: 12px; overflow: auto; font-size: 12px; }
  #form { display: flex; gap: 6px; margin-bottom: 10px; }
  #signal { flex: 1; padding: 6px; }
  pre { white-space: pre-wrap; word-break: break-word; background: #f6f6f6; padding: 8px; }
  #log div { border-bottom: 1px solid #eee; padding: 2px 0; }
  h3 { margin: 12px 0 4px; font-size: 12px; text-transform: uppercase; color: #666; }
</style>
</head>
<body>
<div id="cy"></div>
<div id="side">
  <div id="form">
    <input id="signal" placeholder="raw content de la InputSignal..." />
    <button onclick="submitSignal()">investigar</button>
  </div>
  <h3>nodo</h3>
  <pre id="detail">(click en un nodo)</pre>
  <h3>eventos</h3>
  <div id="log"></div>
</div>
<script>
const SHAPES = { InputSignal: "round-rectangle", Case: "star", Hypothesis: "hexagon",
                 Investigation: "ellipse", Evidence: "triangle", Verdict: "diamond" };
const COLORS = { InputSignal: "#3182ce", Case: "#805ad5", Hypothesis: "#6b46c1",
                 Investigation: "#d69e2e", Evidence: "#38a169", Verdict: "#e53e3e" };

const cy = cytoscape({
  container: document.getElementById("cy"),
  style: [
    { selector: "node", style: {
        label: "data(text)", "font-size": 9, "text-wrap": "wrap", "text-max-width": 120,
        "text-valign": "bottom", "text-margin-y": 4, width: 36, height: 36,
        "background-color": "#999" } },
    { selector: "node[status='refuted']", style: { opacity: 0.45, "border-width": 3, "border-color": "#e53e3e" } },
    { selector: "node[status='skipped']", style: { opacity: 0.4, "border-width": 2, "border-style": "dashed" } },
    { selector: "node[claim='claimed']", style: { "border-width": 3, "border-color": "#f6ad55" } },
    { selector: "edge", style: {
        label: "data(kind)", "font-size": 7, color: "#666", width: 1.5,
        "curve-style": "bezier", "target-arrow-shape": "triangle", "arrow-scale": 0.8 } },
    { selector: "edge[kind='SUGGESTS']", style: {
        "line-color": "#e53e3e", "target-arrow-color": "#e53e3e", "line-style": "dashed", width: 2.5 } },
  ],
});

let pendingEdges = [];
let layoutTimer = null;
function relayout() {
  clearTimeout(layoutTimer);
  layoutTimer = setTimeout(() => cy.layout({
    // hierarchical: the case graph IS a tree (signal -> case -> hypothesis ->
    // investigation -> evidence); layers never overlap, unlike force layouts
    name: "breadthfirst",
    directed: true,
    roots: 'node[type="InputSignal"]',
    spacingFactor: 1.15,
    padding: 40,
    animate: false,
  }).run(), 250);
}

function nodeText(label, props) {
  // short labels: one truncated line (the full text lives in the click panel);
  // long multi-line labels are what made the graph unreadable
  const body = props.description || props.content || props.objective || props.raw_content || props.kind || "";
  const short = String(body).slice(0, 42) + (String(body).length > 42 ? "..." : "");
  return label + "\\n" + short;
}

function upsertNode(label, props) {
  const data = { id: props.id, type: label, text: nodeText(label, props),
                 status: props.status || "", claim: props.claim_state || "", props: props };
  const existing = cy.getElementById(props.id);
  if (existing.length) { existing.data(data); }
  else {
    const el = cy.add({ group: "nodes", data: data });
    el.style({ shape: SHAPES[label] || "ellipse", "background-color": COLORS[label] || "#999" });
    flushPendingEdges();
  }
  relayout();
}

function addEdge(source, kind, target) {
  const id = source + "-" + kind + "-" + target;
  if (cy.getElementById(id).length) return;
  if (!cy.getElementById(source).length || !cy.getElementById(target).length) {
    pendingEdges.push([source, kind, target]);  // endpoint not drawn yet: retry later
    return;
  }
  cy.add({ group: "edges", data: { id: id, source: source, target: target, kind: kind } });
  relayout();
}

function flushPendingEdges() {
  const retry = pendingEdges; pendingEdges = [];
  retry.forEach(([s, k, t]) => addEdge(s, k, t));
}

function logLine(text) {
  const log = document.getElementById("log");
  const div = document.createElement("div");
  div.textContent = text;
  log.prepend(div);
  while (log.childElementCount > 60) log.lastChild.remove();
}

cy.on("tap", "node", (event) => {
  document.getElementById("detail").textContent =
    JSON.stringify(event.target.data("props"), null, 2);
});

const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = (message) => {
  const msg = JSON.parse(message.data);
  if (msg.type === "snapshot") {
    msg.nodes.forEach((n) => upsertNode(n.label, n.props));
    msg.edges.forEach((e) => addEdge(e.source, e.type, e.target));
    logLine(`snapshot: ${msg.nodes.length} nodos, ${msg.edges.length} aristas`);
  } else if (msg.type === "node_created" || msg.type === "node_updated") {
    if (msg.props) upsertNode(msg.node_type, msg.props);
    logLine(`${msg.type} ${msg.node_type} ${(msg.node_id || "").slice(0, 8)}`);
  } else if (msg.type === "edge_created") {
    addEdge(msg.from_id, msg.edge_type, msg.to_id);
    logLine(`edge ${msg.edge_type}`);
  }
};

async function submitSignal() {
  const input = document.getElementById("signal");
  if (!input.value.trim()) return;
  try {
    const res = await fetch("/signal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: input.value }),
    });
    const body = await res.json();
    logLine(`>> senal enviada (${(body.id || "?").slice(0, 8)}) - investigando...`);
    input.value = "";
  } catch (err) {
    logLine(">> ERROR enviando la senal: " + err);
  }
}

// Enter also submits (not only the button)
document.getElementById("signal").addEventListener("keydown", (e) => {
  if (e.key === "Enter") submitSignal();
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    # pipeline logs (roles, claims, failures) share the console with uvicorn's
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # reload=True: picks up code changes automatically (no more stale-page confusion)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
