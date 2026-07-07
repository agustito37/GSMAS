from pathlib import Path

from core.tools.base import Tool


class LogQueryTool(Tool):
    """Keyword search over the security telemetry (a JSONL file: one event per line).
    The demo backend is a flat file; a real deployment would swap this class for a
    SIEM query without touching any role (the contract is the catalog's)."""

    name = "log_query"
    description = (
        "Search the security telemetry logs (authentication, VPN, file access, "
        "network, EDR process events, helpdesk tickets, HR records) for entries "
        "containing the given text. Case-insensitive substring match over the whole "
        "entry. Good queries: a username, an IP, a hostname, an event type. Returns "
        "the matching log lines (max 20)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "text to search for, e.g. 'jdoe', '203.0.113.9', 'FS-03', 'mfa'",
            }
        },
        "required": ["query"],
    }

    def __init__(self, telemetry_path: str) -> None:
        self._path = Path(telemetry_path)

    async def run(self, query: str) -> str:
        if not self._path.exists():
            return f"error: telemetry file not found at {self._path}"
        needle = query.lower()
        hits = [
            line.strip()
            for line in self._path.read_text().splitlines()
            if line.strip() and needle in line.lower()
        ]
        if not hits:
            return f"no log entries match '{query}'"
        return "\n".join(hits[-20:])
