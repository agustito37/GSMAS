// <feedback-panel> — human feedback on a Verdict. Only shows the controls for a Verdict
// that has no feedback yet; posting it triggers the roles' retrospectives server-side.
import { on, sendFeedback, log } from "../bus.js";

class FeedbackPanel extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `<h3>feedback</h3><div id="box">(seleccioná un Verdict)</div>`;
    this._box = this.querySelector("#box");
    on("node:select", ({ type, props }) => this._render(type, props));
    on("graph:snapshot", () => this._render(null, {}));
  }

  _render(type, props) {
    if (type !== "Verdict") {
      this._box.textContent = "(seleccioná un Verdict)";
      return;
    }
    if (props.feedback) {
      this._box.textContent = "feedback: " + props.feedback;
      return;
    }
    this._box.innerHTML = "";
    ["correct", "incorrect", "partial"].forEach((f) => {
      const btn = document.createElement("button");
      btn.textContent = f;
      btn.style.marginRight = "6px";
      btn.onclick = () => this._send(props.id, f);
      this._box.appendChild(btn);
    });
  }

  async _send(verdictId, feedback) {
    try {
      await sendFeedback(verdictId, feedback);
      this._box.textContent = "feedback: " + feedback;
      log(`>> feedback ${feedback} en verdict ${verdictId.slice(0, 8)} - retrospectivas disparadas`);
    } catch (err) {
      log(">> ERROR enviando feedback: " + err);
    }
  }
}

customElements.define("feedback-panel", FeedbackPanel);
