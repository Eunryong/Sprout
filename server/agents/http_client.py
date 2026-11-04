"""Chat-completions client for the real models. ASSUMPTION (to confirm against the hackathon API docs): an
OpenAI-style `POST {base_url}/chat/completions` with `Authorization: Bearer <key>`, `messages`, and
`choices[0].message.content` in the reply. Only this file knows the wire format; everything else sees LLMClient."""
import json
import time
import urllib.error
import urllib.request
from typing import Callable

from pydantic import BaseModel

from .client import ClientError
from .secrets import redact

RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def urllib_transport(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, str]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def extract_json(text: str) -> dict:
    """The model's reply -> a JSON object. Tolerates ```json fences and prose around the object."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        t = t.rsplit("```", 1)[0]
    try:
        out = json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j <= i:
            raise ClientError("reply has no JSON object") from None
        try:
            out = json.loads(t[i:j + 1])
        except json.JSONDecodeError as e:
            raise ClientError(f"reply is not valid JSON ({e.msg})") from None
    if not isinstance(out, dict):
        raise ClientError("reply JSON is not an object")
    return out


class ChatJSONClient:
    def __init__(self, *, name: str, model: str, base_url: str, api_key: str, timeout: float = 30.0,
                 temperature: float = 0.8, max_tokens: int = 4000, retries: int = 2, json_mode: bool = True,
                 reasoning_effort: str | None = None, transport: Callable = urllib_transport, sleep: Callable = time.sleep):
        if not base_url or not api_key:
            raise ClientError(f"{name}: base_url and api key are required")
        self.name, self.model, self.api_key = name, model, api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout, self.temperature, self.max_tokens, self.retries = timeout, temperature, max_tokens, retries
        self.transport, self.sleep = transport, sleep
        self.reasoning_effort = reasoning_effort  # these are reasoning models: thinking tokens count against max_tokens
        self.json_mode = json_mode  # sent as response_format; switched off for good if the server rejects it
        self.tokens_in = self.tokens_out = self.tokens_reasoning = self.requests = 0

    def _err(self, msg: str) -> ClientError:
        return ClientError(f"{self.name}: {redact(msg, (self.api_key,))}")

    def complete_json(self, *, agent: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        payload = {
            "model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system + "\n\nReply with ONE JSON object that matches this JSON Schema, "
                                                        "and nothing else:\n" + json.dumps(schema.model_json_schema())},
                {"role": "user", "content": user}]}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        last = "no attempt"
        for attempt in range(self.retries + 1):
            if attempt:
                self.sleep(min(8.0, 1.0 * 2 ** (attempt - 1)))
            self.requests += 1
            if self.json_mode:
                payload["response_format"] = {"type": "json_object"}
            else:
                payload.pop("response_format", None)
            body = json.dumps(payload).encode()
            try:
                status, text = self.transport(self.url, headers, body, self.timeout)
            except Exception as e:  # noqa: BLE001 - timeouts, DNS, TLS: all worth a retry
                last = f"{type(e).__name__}: {e}"
                continue
            if status in RETRY_STATUS:
                last = f"HTTP {status}"
                continue
            if status == 400 and self.json_mode and "response_format" in text:
                self.json_mode = False  # this server has no JSON mode: ask in the prompt only
                last = "HTTP 400 (response_format not supported)"
                continue
            if status != 200:
                raise self._err(f"HTTP {status}: {text[:200]}")
            try:
                resp = json.loads(text)
                choice = resp["choices"][0]
                content = choice["message"]["content"]
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                raise self._err("unexpected response shape") from None
            usage = resp.get("usage") or {}
            self.tokens_in += int(usage.get("prompt_tokens", 0) or 0)
            self.tokens_out += int(usage.get("completion_tokens", 0) or 0)
            self.tokens_reasoning += int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
            if not (content or "").strip() and choice.get("finish_reason") == "length":
                raise self._err(f"max_tokens={self.max_tokens} was used up by reasoning before any answer (raise it, "
                                f"or set reasoning_effort)")
            try:
                return schema.model_validate(extract_json(content or ""))
            except ClientError as e:
                raise self._err(str(e)) from None
            except Exception as e:  # noqa: BLE001 - pydantic ValidationError
                raise self._err(f"reply does not match the schema: {str(e)[:200]}") from None
        raise self._err(f"gave up after {self.retries + 1} tries ({last})")
