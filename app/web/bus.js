// Coordination layer for the dashboard components: a shared event bus plus the backend
// clients (one WebSocket + fetch helpers). Components never talk to each other directly;
// they emit and listen on `bus`, so each stays self-contained.

export const bus = new EventTarget();

export const emit = (name, detail) => bus.dispatchEvent(new CustomEvent(name, { detail }));
export const on = (name, handler) => bus.addEventListener(name, (e) => handler(e.detail));
export const log = (text) => emit("log:line", { text });

let sock = null;
let workspace = "default";

export function getWorkspace() {
  return workspace;
}

// Open the WebSocket and fan its messages onto the bus. The server sends a snapshot of
// the current workspace on connect and on every switch, then live mutations. Auto-
// reconnects on close, so the dashboard survives server reloads (reload=True restarts
// uvicorn and drops every socket on each code change during development).
export function connect() {
  sock = new WebSocket(`ws://${location.host}/ws`);
  sock.onopen = () => {
    emit("socket:open", {});
    // re-scope to whatever workspace the user was viewing (the server starts at default)
    if (workspace !== "default") sock.send(JSON.stringify({ workspace }));
  };
  sock.onmessage = (message) => {
    const msg = JSON.parse(message.data);
    if (msg.type === "snapshot") emit("graph:snapshot", msg);
    else emit("graph:event", msg);
  };
  sock.onclose = () => {
    log(">> socket cerrado - reconectando...");
    setTimeout(connect, 1000);
  };
}

// Re-scope the view to a workspace: tell the server (it re-snapshots), and announce it.
export function switchWorkspace(next) {
  workspace = next;
  if (sock && sock.readyState === WebSocket.OPEN) sock.send(JSON.stringify({ workspace }));
  emit("workspace:change", { workspace });
}

export async function listWorkspaces() {
  const res = await fetch("/workspaces");
  return (await res.json()).workspaces;
}

export async function deleteWorkspace(name) {
  const res = await fetch(`/workspaces/${encodeURIComponent(name)}`, { method: "DELETE" });
  return res.json();
}

export async function submitSignal(content) {
  const res = await fetch("/signal", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, workspace }),
  });
  return res.json();
}

export async function sendFeedback(verdictId, feedback) {
  await fetch(`/verdict/${verdictId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ feedback }),
  });
}
