// <graph-view> — the live Cytoscape graph of ONE workspace. Renders snapshots and
// live mutations from the bus, and emits node:select when a node is tapped. Cytoscape
// is loaded globally from the CDN (see index.html).
//
// Color language (consistent everywhere): green = positive (supports, corroborated,
// confirmed), red = negative (contradicts, refuted), amber = in progress, gray = neutral
// / inactive / unresolved / retired, blue = input / applied, violet = generative
// (case, hypothesis, suggests), teal = knowledge (skill). Shapes encode the node TYPE;
// borders and opacity encode STATE - so a base color never contradicts a state.
import { on, emit } from "../bus.js";

const STYLE = [
  { selector: "node", style: {
      label: "data(text)", "font-size": 9, "text-wrap": "wrap", "text-max-width": 120,
      "text-valign": "bottom", "text-margin-y": 4, width: 36, height: 36,
      shape: "ellipse", "background-color": "#94a3b8" } },

  // type -> shape + base color
  { selector: "node[type='InputSignal']", style: { shape: "round-rectangle", "background-color": "#3182ce" } },
  { selector: "node[type='Case']", style: { shape: "star", "background-color": "#7c3aed" } },
  { selector: "node[type='Hypothesis']", style: { shape: "hexagon", "background-color": "#a78bfa" } },
  { selector: "node[type='Investigation']", style: { shape: "ellipse", "background-color": "#d69e2e" } },
  { selector: "node[type='Evidence']", style: { shape: "triangle", "background-color": "#64748b" } },
  { selector: "node[type='Verdict']", style: { shape: "diamond", "background-color": "#334155" } },
  { selector: "node[type='Workspace']", style: { shape: "rectangle", "background-color": "#1e293b" } },
  { selector: "node[type='Role']", style: { shape: "vee", "background-color": "#4f46e5" } },
  { selector: "node[type='LTM']", style: { shape: "barrel", "background-color": "#818cf8" } },
  { selector: "node[type='Skill']", style: { shape: "tag", "background-color": "#0d9488" } },

  // state -> border / opacity (listed after type so they win)
  { selector: "node[status='confirmed']", style: { "border-width": 3, "border-color": "#38a169" } },
  { selector: "node[status='refuted']", style: { opacity: 0.4, "border-width": 3, "border-color": "#e53e3e" } },
  { selector: "node[status='retired']", style: { opacity: 0.4, "background-color": "#94a3b8" } },
  { selector: "node[status='skipped']", style: { opacity: 0.4, "border-width": 2, "border-style": "dashed" } },
  { selector: "node[claim='claimed']", style: { "border-width": 3, "border-color": "#f6ad55" } },
  { selector: "node[kind='unresolved']", style: { "background-color": "#94a3b8" } },

  // edges: neutral by default; positive green, negative red, generative violet
  { selector: "edge", style: {
      label: "data(kind)", "font-size": 7, color: "#666", width: 1.5,
      "line-color": "#cbd5e1", "target-arrow-color": "#cbd5e1",
      "curve-style": "bezier", "target-arrow-shape": "triangle", "arrow-scale": 0.8 } },
  { selector: "edge[kind='SUPPORTS']", style: { "line-color": "#38a169", "target-arrow-color": "#38a169" } },
  { selector: "edge[kind='CONTRADICTS']", style: { "line-color": "#e53e3e", "target-arrow-color": "#e53e3e" } },
  { selector: "edge[kind='SUGGESTS']", style: {
      "line-color": "#7c3aed", "target-arrow-color": "#7c3aed", "line-style": "dashed", width: 2 } },
  // learning edges (visible once a retrospective has run)
  { selector: "edge[kind='HAS_SKILL']", style: { "line-color": "#0d9488", "target-arrow-color": "#0d9488" } },
  { selector: "edge[kind='APPLIED']", style: {
      "line-color": "#3182ce", "target-arrow-color": "#3182ce", "line-style": "dotted" } },
  // vitality / bookkeeping edges to the Case (CORROBORATED_BY, REFUTED_BY, RETROSPECTED)
  // exist in the graph for traceability, but are HIDDEN here: they cross-link the memory
  // back to the investigation and tangle the layout. The vitality still lives in the data.
  { selector: "edge[kind='CORROBORATED_BY']", style: { display: "none" } },
  { selector: "edge[kind='REFUTED_BY']", style: { display: "none" } },
  { selector: "edge[kind='RETROSPECTED']", style: { display: "none" } },
];

class GraphView extends HTMLElement {
  connectedCallback() {
    this._pending = [];
    this._layoutTimer = null;
    this._pinned = {}; // nodes the user dragged: id -> position, kept across re-layouts
    // dagre gives a real layered DAG layout (proper subtree packing and edge routing);
    // fall back to the built-in breadthfirst if the extension did not load
    this._hasDagre = false;
    if (window.cytoscapeDagre) {
      try { cytoscape.use(window.cytoscapeDagre); } catch (e) { /* already registered */ }
      this._hasDagre = true;
    }
    this._cy = cytoscape({ container: this, style: STYLE });

    this._cy.on("tap", "node", (event) => {
      emit("node:select", { type: event.target.data("type"), props: event.target.data("props") });
    });

    // pin a node once the user actually drags it (compare grab vs release, so a plain
    // tap-to-inspect does not pin): a pinned node stays put when later nodes lay out
    this._cy.on("grab", "node", (e) => { this._grabPos = { ...e.target.position() }; });
    this._cy.on("free", "node", (e) => {
      const p = e.target.position();
      if (this._grabPos &&
          (Math.abs(p.x - this._grabPos.x) > 2 || Math.abs(p.y - this._grabPos.y) > 2)) {
        this._pinned[e.target.id()] = { ...p };
      }
    });

    on("graph:snapshot", (msg) => this._snapshot(msg));
    on("graph:event", (msg) => this._event(msg));
  }

  _snapshot(msg) {
    this._cy.elements().remove(); // a snapshot is the full state of ONE workspace
    this._pending = [];
    this._pinned = {}; // a fresh workspace starts from a clean layout
    msg.nodes.forEach((n) => this._upsert(n.label, n.props));
    msg.edges.forEach((e) => this._addEdge(e.source, e.type, e.target));
    emit("log:line", {
      text: `snapshot ${msg.workspace || ""}: ${msg.nodes.length} nodos, ${msg.edges.length} aristas`,
    });
  }

  _event(msg) {
    if (msg.type === "node_created" || msg.type === "node_updated") {
      if (msg.props) this._upsert(msg.node_type, msg.props);
      emit("log:line", { text: `${msg.type} ${msg.node_type} ${(msg.node_id || "").slice(0, 8)}` });
    } else if (msg.type === "edge_created") {
      this._addEdge(msg.from_id, msg.edge_type, msg.to_id);
      emit("log:line", { text: `edge ${msg.edge_type}` });
    }
  }

  _nodeText(label, props) {
    // keep graph labels SHORT (type + a hint): 28 nodes in a rank overlap otherwise.
    // the full text is one tap away in <node-detail>.
    const body =
      props.description || props.summary || props.content || props.name ||
      props.objective || props.raw_content || props.kind || "";
    const short = String(body).slice(0, 20) + (String(body).length > 20 ? "..." : "");
    return short ? label + "\n" + short : label;
  }

  _upsert(label, props) {
    // all styling lives in the stylesheet, keyed by these data fields (type/status/
    // claim/kind), so state can override the base color without inline conflicts
    const data = {
      id: props.id, type: label, text: this._nodeText(label, props),
      status: props.status || "", claim: props.claim_state || "",
      kind: props.kind || "", props: props,
    };
    const existing = this._cy.getElementById(props.id);
    if (existing.length) {
      existing.data(data); // just restyle in place: no re-layout, so positions stay put
    } else {
      this._cy.add({ group: "nodes", data: data });
      this._flushPending();
      this._relayout(); // only a NEW node reshuffles the tree
    }
  }

  _addEdge(source, kind, target) {
    const id = source + "-" + kind + "-" + target;
    if (this._cy.getElementById(id).length) return;
    if (!this._cy.getElementById(source).length || !this._cy.getElementById(target).length) {
      this._pending.push([source, kind, target]); // endpoint not drawn yet: retry later
      return;
    }
    this._cy.add({ group: "edges", data: { id, source, target, kind } });
    this._relayout();
  }

  _flushPending() {
    const retry = this._pending;
    this._pending = [];
    retry.forEach(([s, k, t]) => this._addEdge(s, k, t));
  }

  _relayout() {
    clearTimeout(this._layoutTimer);
    this._layoutTimer = setTimeout(() => {
      // the case graph is a layered DAG (signal -> case -> hypothesis -> investigation
      // -> evidence, plus workspace -> role -> ltm -> skill). dagre packs the subtrees
      // and routes the cross edges (evidence -> hypothesis) far better than breadthfirst.
      const opts = this._hasDagre
        ? {
            name: "dagre",
            rankDir: "TB",
            nodeSep: 40, // horizontal gap between siblings in a rank
            rankSep: 90, // vertical gap between the type layers
            edgeSep: 12,
            animate: false,
            padding: 40,
          }
        : {
            name: "breadthfirst",
            directed: true,
            roots: 'node[type="Workspace"], node[type="InputSignal"]',
            spacingFactor: 1.8,
            avoidOverlap: true,
            padding: 40,
            animate: false,
          };
      this._cy.layout(opts).run();
      // re-apply user-dragged positions so a new node never undoes a manual move
      for (const [id, pos] of Object.entries(this._pinned)) {
        const el = this._cy.getElementById(id);
        if (el.length) el.position(pos);
      }
    }, 250);
  }
}

customElements.define("graph-view", GraphView);
