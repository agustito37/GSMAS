// <node-detail> — the full properties of the tapped node (cleared on workspace switch).
import { on } from "../bus.js";

class NodeDetail extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `<h3>nodo</h3><pre id="detail">(click en un nodo)</pre>`;
    this._pre = this.querySelector("#detail");
    on("node:select", ({ props }) => {
      this._pre.textContent = JSON.stringify(props, null, 2);
    });
    on("graph:snapshot", () => {
      this._pre.textContent = "(click en un nodo)";
    });
  }
}

customElements.define("node-detail", NodeDetail);
