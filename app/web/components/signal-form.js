// <signal-form> — submit an InputSignal into the currently selected workspace.
import { submitSignal, switchWorkspace, getWorkspace, log } from "../bus.js";

class SignalForm extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `
      <input id="signal" placeholder="raw content de la InputSignal..." />
      <button id="go">investigar</button>
    `;
    this._input = this.querySelector("#signal");
    this.querySelector("#go").addEventListener("click", () => this._submit());
    this._input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") this._submit();
    });
  }

  async _submit() {
    const content = this._input.value.trim();
    if (!content) return;
    try {
      const body = await submitSignal(content);
      log(`>> senal enviada a ${getWorkspace()} (${(body.id || "?").slice(0, 8)}) - investigando...`);
      this._input.value = "";
      switchWorkspace(getWorkspace()); // re-snapshot: pull in the just-reified roles/LTMs
    } catch (err) {
      log(">> ERROR enviando la senal: " + err);
    }
  }
}

customElements.define("signal-form", SignalForm);
