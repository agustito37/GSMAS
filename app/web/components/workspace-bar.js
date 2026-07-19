// <workspace-bar> — pick a workspace to view, or create a new one. Switching re-scopes
// the whole view (the server re-snapshots that workspace).
import { listWorkspaces, switchWorkspace, getWorkspace, deleteWorkspace, log } from "../bus.js";

class WorkspaceBar extends HTMLElement {
  async connectedCallback() {
    this.innerHTML = `
      <select id="ws"></select>
      <input id="newws" placeholder="new workspace..." />
      <button id="add" title="create workspace">+</button>
      <button id="del" title="delete workspace and all its nodes">🗑</button>
    `;
    this._sel = this.querySelector("#ws");
    this._sel.addEventListener("change", () => this._switch());
    this.querySelector("#add").addEventListener("click", () => this._create());
    this.querySelector("#del").addEventListener("click", () => this._delete());
    await this.reload();
  }

  async reload() {
    const current = this._sel.value || getWorkspace();
    const workspaces = await listWorkspaces();
    this._sel.innerHTML = "";
    workspaces.forEach((w) => {
      const opt = document.createElement("option");
      opt.value = w;
      opt.textContent = w;
      this._sel.appendChild(opt);
    });
    this._sel.value = workspaces.includes(current) ? current : "default";
  }

  _switch() {
    switchWorkspace(this._sel.value || "default");
  }

  _create() {
    const input = this.querySelector("#newws");
    const name = input.value.trim();
    if (!name) return;
    if (![...this._sel.options].some((o) => o.value === name)) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      this._sel.appendChild(opt);
    }
    this._sel.value = name;
    input.value = "";
    this._switch(); // the workspace appears in the graph once a signal is submitted to it
  }

  async _delete() {
    const ws = this._sel.value || "default";
    if (!confirm(`Delete workspace "${ws}" and ALL its nodes? This cannot be undone.`)) return;
    try {
      const body = await deleteWorkspace(ws);
      log(`>> workspace ${ws} deleted (${body.deleted} nodes)`);
      await this.reload(); // the deleted workspace is gone from the list -> falls back to default
      this._switch();
    } catch (err) {
      log(">> ERROR eliminando workspace: " + err);
    }
  }
}

customElements.define("workspace-bar", WorkspaceBar);
