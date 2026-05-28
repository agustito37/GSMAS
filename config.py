import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH, override=False)


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val or val.startswith("sk-replace"):
        raise RuntimeError(f"Missing or placeholder env var: {name}. Set it in {ENV_PATH}.")
    return val


OPENAI_API_KEY = _require("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID") or None
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID") or None
