// <agent-activity> — live tracing of the ephemeral agents: which role is working on
// what, right now. Agents are not graph nodes, so this panel is the only window into
// the swarm's activity. Fed by agent_started / agent_finished / agent_failed events.
import { on } from "../bus.js";

// one colour per role (matches the graph's language: violet=generative, amber=planning,
// blue=gathering, teal=concluding); unknown roles fall back to slate
const ROLE_COLOR = {
  proposer: "#7c3aed",
  planner: "#d69e2e",
  investigator: "#3182ce",
  aggregator: "#0d9488",
};

// the reaction each agent runs (the execute's name), to a readable verb
const ACTION_LABEL = {
  open_case: "opening case",
  triage_evidence: "triaging",
  plan: "planning",
  investigate: "investigating",
  conclude: "concluding",
  retrospect: "retrospective",
};

const LABEL_FIELDS = ["description", "objective", "raw_content", "content", "summary"];

class AgentActivity extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `<h3>active agents <span id="acount"></span></h3><div id="agents"></div>`;
    this._list = this.querySelector("#agents");
    this._count = this.querySelector("#acount");
    this._active = new Map(); // work_id -> {role, type, label}
    on("agent:event", (msg) => this._event(msg));
    // a snapshot means a workspace switch or a reconnect: the in-flight set is unknown,
    // so start clean and let new agent events repopulate it
    on("graph:snapshot", () => this._reset());
    this._render();
  }

  _reset() {
    this._active.clear();
    this._render();
  }

  _event(msg) {
    if (!msg.node_id) return;
    // key by role+action+work, not work alone: several agents can run on the SAME node
    // (the four roles each retrospect the same Case), so keying by work would collapse
    // them into one and only the last would show
    const key = `${msg.role}:${msg.action}:${msg.node_id}`;
    if (msg.type === "agent_started") {
      this._active.set(key, {
        role: msg.role || "?",
        action: msg.action || "",
        type: msg.node_type || "",
        label: this._label(msg.props),
      });
    } else {
      this._active.delete(key); // finished or failed: the agent is gone
    }
    this._render();
  }

  _label(props) {
    if (!props) return "";
    for (const field of LABEL_FIELDS) {
      if (props[field]) {
        const text = String(props[field]);
        return text.slice(0, 44) + (text.length > 44 ? "..." : "");
      }
    }
    return "";
  }

  _render() {
    this._count.textContent = this._active.size ? `(${this._active.size})` : "";
    if (this._active.size === 0) {
      this._list.innerHTML = `<div class="agent-idle">idle</div>`;
      return;
    }
    this._list.innerHTML = "";
    for (const agent of this._active.values()) {
      const row = document.createElement("div");
      row.className = "agent-row";
      const color = ROLE_COLOR[agent.role] || "#64748b";
      const action = ACTION_LABEL[agent.action] || agent.action;
      const on = agent.label ? `${agent.type}: ${agent.label}` : agent.type;
      row.innerHTML =
        `<span class="role-badge" style="background:${color}">${agent.role}</span>` +
        `<span class="agent-work"><b>${action}</b> · ${on}</span>`;
      this._list.appendChild(row);
    }
  }
}

customElements.define("agent-activity", AgentActivity);
