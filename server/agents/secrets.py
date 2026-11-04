"""API keys live in the environment (or a git-ignored .env.local / .env), never in config files, saves, logs or events."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECRET_ENVS = ("MUSE_API_KEY",)


def load_dotenv(path: Path | None = None) -> list[str]:
    """Read KEY=VALUE lines into os.environ without overriding what is already set (default files: .env.local, then
    .env). Returns the names that were loaded (never the values)."""
    paths = [path] if path else [ROOT / ".env.local", ROOT / ".env"]
    loaded = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v
                loaded.append(k)
    return loaded


def redact(text: str | None, extra: tuple[str, ...] = ()) -> str | None:
    """Replace any known secret value in `text` (error messages, log rows)."""
    if not text:
        return text
    for name in SECRET_ENVS:
        v = os.environ.get(name)
        if v and len(v) >= 6:
            text = text.replace(v, "***")
    for v in extra:
        if v and len(v) >= 6:
            text = text.replace(v, "***")
    return text
