// <event-log> — a rolling feed of what happened (mutations, submissions, feedback).
import { on } from "../bus.js";

const MAX_LINES = 60;

class EventLog extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `<h3>events</h3><div id="log"></div>`;
    this._log = this.querySelector("#log");
    on("log:line", ({ text }) => this._line(text));
  }

  _line(text) {
    const div = document.createElement("div");
    div.textContent = text;
    this._log.prepend(div);
    while (this._log.childElementCount > MAX_LINES) this._log.lastChild.remove();
  }
}

customElements.define("event-log", EventLog);
