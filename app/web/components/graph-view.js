// <graph-view> — the live Cytoscape graph of ONE workspace. Renders snapshots and
// live mutations from the bus, and emits node:select when a node is tapped. Cytoscape
// is loaded globally from the CDN (see index.html).
import { on, emit } from "../bus.js";

const SHAPES = {
  InputSignal: "round-rectangle", Case: "star", Hypothesis: "hexagon",
  Investigation: "ellipse", Evidence: "triangle", Verdict: "diamond",
};
const COLORS = {
  InputSignal: "#3182ce", Case: "#805ad5", Hypothesis: "#6b46c1",
  Investigation: "#d69e2e", Evidence: "#38a169", Verdict: "#e53e3e",
};

class GraphView extends HTMLElement {
  connectedCallback() {
    this._pending = [];
    this._layoutTimer = null;
    this._cy = cytoscape({
      container: this,
      style: [
        { selector: "node", style: {
            label: "data(text)", "font-size": 9, "text-wrap": "wrap", "text-max-width": 120,
            "text-valign": "bottom", "text-margin-y": 4, width: 36, height: 36,
            "background-color": "#999" } },
        { selector: "node[status='refuted']", style: {
            opacity: 0.45, "border-width": 3, "border-color": "#e53e3e" } },
        { selector: "node[status='skipped']", style: {
            opacity: 0.4, "border-width": 2, "border-style": "dashed" } },
        { selector: "node[claim='claimed']", style: {
            "border-width": 3, "border-color": "#f6ad55" } },
        { selector: "edge", style: {
            label: "data(kind)", "font-size": 7, color: "#666", width: 1.5,
            "curve-style": "bezier", "target-arrow-shape": "triangle", "arrow-scale": 0.8 } },
        { selector: "edge[kind='SUGGESTS']", style: {
            "line-color": "#e53e3e", "target-arrow-color": "#e53e3e",
            "line-style": "dashed", width: 2.5 } },
      ],
    });

    this._cy.on("tap", "node", (event) => {
      emit("node:select", { type: event.target.data("type"), props: event.target.data("props") });
    });

    on("graph:snapshot", (msg) => this._snapshot(msg));
    on("graph:event", (msg) => this._event(msg));
  }

  _snapshot(msg) {
    this._cy.elements().remove(); // a snapshot is the full state of ONE workspace
    this._pending = [];
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
    const body =
      props.description || props.content || props.objective || props.raw_content || props.kind || "";
    const short = String(body).slice(0, 42) + (String(body).length > 42 ? "..." : "");
    return label + "\n" + short;
  }

  _upsert(label, props) {
    const data = {
      id: props.id, type: label, text: this._nodeText(label, props),
      status: props.status || "", claim: props.claim_state || "", props: props,
    };
    const existing = this._cy.getElementById(props.id);
    if (existing.length) {
      existing.data(data);
    } else {
      const el = this._cy.add({ group: "nodes", data: data });
      el.style({ shape: SHAPES[label] || "ellipse", "background-color": COLORS[label] || "#999" });
      this._flushPending();
    }
    this._relayout();
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
    this._layoutTimer = setTimeout(() => this._cy.layout({
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
}

customElements.define("graph-view", GraphView);
