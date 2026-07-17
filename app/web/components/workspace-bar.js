// <workspace-bar> — pick a workspace to view, or create a new one. Switching re-scopes
// the whole view (the server re-snapshots that workspace).
import { listWorkspaces, switchWorkspace, getWorkspace } from "../bus.js";

class WorkspaceBar extends HTMLElement {
  async connectedCallback() {
    this.innerHTML = `
      <select id="ws"></select>
      <input id="newws" placeholder="nuevo workspace..." />
      <button id="add">+</button>
    `;
    this._sel = this.querySelector("#ws");
    this._sel.addEventListener("change", () => this._switch());
    this.querySelector("#add").addEventListener("click", () => this._create());
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
}

customElements.define("workspace-bar", WorkspaceBar);
