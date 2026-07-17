// Entry point: register every component (each file calls customElements.define on import)
// and open the socket. The custom elements already in index.html upgrade as they are
// defined, so their listeners are wired before connect() delivers the first snapshot.
import { connect } from "./bus.js";
import "./components/workspace-bar.js";
import "./components/signal-form.js";
import "./components/graph-view.js";
import "./components/node-detail.js";
import "./components/feedback-panel.js";
import "./components/event-log.js";

connect();
