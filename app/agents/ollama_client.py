"""
Ollama Client — built for Qwen3 4B reality.
The model often returns empty strings or plain text instead of JSON.
This client handles every failure mode gracefully.
"""

import json
import re
import logging
import asyncio
from typing import TypeVar, Type

import httpx
from pydantic import BaseModel

from app.config import settings
from app.core.exceptions import LLMError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class OllamaTokenTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0

    def record(self, prompt_tokens: int, completion_tokens: int):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_requests += 1

    def summary(self) -> str:
        total = self.total_prompt_tokens + self.total_completion_tokens
        return (
            f"ollama[reqs={self.total_requests}, "
            f"prompt={self.total_prompt_tokens}, "
            f"completion={self.total_completion_tokens}, "
            f"total={total}]"
        )


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ):
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.ollama_model
        self._timeout = timeout or settings.ollama_timeout
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)
        self.token_usage = OllamaTokenTracker()

        logger.info(
            "OllamaClient | model=%s | url=%s | timeout=%ds",
            self._model, self._base_url, self._timeout,
        )

    async def _call_ollama(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 8192,
        force_json: bool = False,
    ) -> str:
        """Raw Ollama API call."""
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 16384,
            },
        }

        # Force JSON output format when available
        if force_json:
            payload["format"] = "json"

        async with self._semaphore:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
                    resp = await client.post(
                        f"{self._base_url}/api/chat",
                        json=payload,
                    )

                    if resp.status_code != 200:
                        raise LLMError(f"Ollama HTTP {resp.status_code}: {resp.text[:300]}")

                    data = resp.json()
                    content = data.get("message", {}).get("content", "")

                    prompt_tokens = data.get("prompt_eval_count", 0)
                    completion_tokens = data.get("eval_count", 0)
                    self.token_usage.record(prompt_tokens, completion_tokens)

                    total_ms = data.get("total_duration", 0) / 1_000_000
                    logger.info(
                        "Ollama OK | prompt_tok=%d | comp_tok=%d | time=%.1fs | json_mode=%s",
                        prompt_tokens, completion_tokens, total_ms / 1000, force_json,
                    )

                    # Strip think blocks
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

                    # DEBUG: Log first 300 chars of actual response
                    if content:
                        logger.debug("Ollama raw response (first 300): %.300s", content)
                    else:
                        logger.warning("Ollama returned EMPTY response (0 chars after stripping)")

                    return content

            except httpx.TimeoutException:
                raise LLMError(f"Ollama timeout ({self._timeout}s)")
            except httpx.ConnectError:
                raise LLMError(f"Cannot connect to Ollama at {self._base_url}")
            except LLMError:
                raise
            except Exception as e:
                raise LLMError(f"Ollama error: {e}")

    async def chat(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_fast_model: bool = False,
    ) -> str:
        """Chat with Ollama. Retries on empty response."""
        temp = temperature if temperature is not None else settings.groq_temperature
        clean_system = system.replace("/no_think", "").strip()

        messages = [
            {"role": "system", "content": clean_system},
            {"role": "user", "content": prompt},
        ]

        content = await self._call_ollama(messages, temp, max_tokens or 8192)

        # Retry on empty
        if not content.strip():
            logger.warning("Empty response — retry with shorter prompt + lower temp")
            short = prompt[:2000] if len(prompt) > 2000 else prompt
            messages = [
                {"role": "system", "content": "Answer concisely."},
                {"role": "user", "content": short},
            ]
            content = await self._call_ollama(messages, 0.0, max_tokens or 8192)

        return content

    def _extract_json(self, content: str) -> str:
        """Extract JSON from messy model output."""
        if not content or not content.strip():
            return "{}"

        cleaned = content.strip()

        # Markdown fence
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Brace match
        first = cleaned.find("{")
        if first != -1:
            depth = 0
            for i in range(first, len(cleaned)):
                if cleaned[i] == "{":
                    depth += 1
                elif cleaned[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return cleaned[first:i + 1]
            # Unclosed — fix it
            return self._close_truncated_json(cleaned[first:])

        return "{}"

    def _close_truncated_json(self, partial: str) -> str:
        """Fix truncated JSON."""
        partial = partial.rstrip().rstrip(",")

        # Close unclosed string
        in_str = False
        esc = False
        for ch in partial:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
        if in_str:
            partial += '"'

        open_b = partial.count("[") - partial.count("]")
        open_c = partial.count("{") - partial.count("}")
        partial += "]" * max(0, open_b)
        partial += "}" * max(0, open_c)
        return partial

    def _fix_flat_arrays(self, data: dict, schema: dict) -> dict:
        """Fix ["string"] → [{"name": "string", ...}]."""
        props = schema.get("properties", {})
        defs = schema.get("$defs", {})

        for fname, finfo in props.items():
            if fname not in data:
                continue
            val = data[fname]
            if not isinstance(val, list) or not val or not isinstance(val[0], str):
                continue
            if finfo.get("type") != "array":
                continue

            items = finfo.get("items", {})
            if "$ref" in items:
                ref_name = items["$ref"].split("/")[-1]
                items = defs.get(ref_name, items)
            if "anyOf" in items:
                for opt in items["anyOf"]:
                    if "$ref" in opt:
                        items = defs.get(opt["$ref"].split("/")[-1], items)
                        break

            items_props = items.get("properties", {})
            if not items_props:
                continue

            pk = None
            for k, v in items_props.items():
                if v.get("type", "") == "string" or (not v.get("type") and not v.get("$ref")):
                    pk = k
                    break
            if not pk:
                pk = list(items_props.keys())[0]

            fixed = []
            for item in val:
                if isinstance(item, str):
                    obj = {pk: item}
                    for ok, ov in items_props.items():
                        if ok != pk:
                            ot = ov.get("type", "string")
                            obj.setdefault(ok, [] if ot == "array" else (0 if ot == "integer" else ""))
                    fixed.append(obj)
                else:
                    fixed.append(item)
            data[fname] = fixed
            logger.debug("Fixed flat array '%s': %d items", fname, len(fixed))

        return data

    def _parse_text_to_dict(self, text: str, schema: dict) -> dict:
        """
        Last resort: parse plain text response into a dict.
        Looks for field names from the schema in the text.
        """
        props = schema.get("properties", {})
        result = {}

        for fname, finfo in props.items():
            ftype = finfo.get("type", "string")
            default = finfo.get("default", None)

            if default is not None:
                result[fname] = default
                continue

            # Try to find "field_name: value" or "field_name = value" in text
            pattern = rf'(?:"{fname}"|{fname})\s*[:=]\s*(.+?)(?:\n|$)'
            m = re.search(pattern, text, re.IGNORECASE)

            if m:
                raw_val = m.group(1).strip().strip('"').strip("'")
                if ftype == "string":
                    result[fname] = raw_val
                elif ftype == "array":
                    # Try to parse comma-separated
                    items = [x.strip().strip('"').strip("'") for x in raw_val.split(",")]
                    result[fname] = [x for x in items if x]
                elif ftype == "integer":
                    try:
                        result[fname] = int(raw_val)
                    except ValueError:
                        result[fname] = 0
                elif ftype == "boolean":
                    result[fname] = raw_val.lower() in ("true", "yes", "1")
                else:
                    result[fname] = raw_val
            else:
                # Use defaults
                if ftype == "string":
                    if fname in ("summary", "overview", "description"):
                        # Use the whole text as summary
                        clean = re.sub(r"[{}\[\]]", "", text).strip()
                        result[fname] = clean[:500] if clean else ""
                    else:
                        result[fname] = ""
                elif ftype == "array":
                    result[fname] = []
                elif ftype == "integer":
                    result[fname] = 0
                elif ftype == "boolean":
                    result[fname] = False
                elif ftype == "object":
                    result[fname] = {}
                else:
                    result[fname] = None

        return result

    async def structured_chat(
        self,
        prompt: str,
        system: str,
        response_model: Type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_fast_model: bool = False,
    ) -> T:
        """
        Get structured output from Ollama.

        Strategy (4 attempts):
          1. JSON mode (format=json) with full prompt
          2. JSON mode with shorter prompt
          3. Normal mode with minimal prompt
          4. Parse plain text into schema fields
        """
        schema = response_model.model_json_schema()
        example = self._build_example(response_model)
        clean_system = system.replace("/no_think", "").strip()

        fields = list(schema.get("properties", {}).keys())
        fields_str = ", ".join(fields)

        last_content = ""

        # ===== ATTEMPT 1: JSON mode + full prompt =====
        try:
            json_system = (
                f"{clean_system}\n\n"
                f"Respond with a JSON object containing: {fields_str}"
            )
            content = await self._call_ollama(
                messages=[
                    {"role": "system", "content": json_system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature or 0.1,
                max_tokens=max_tokens or 8192,
                force_json=True,
            )
            last_content = content
            if content.strip():
                json_str = self._extract_json(content)
                data = json.loads(json_str)
                data = self._fix_flat_arrays(data, schema)
                result = response_model.model_validate(data)
                logger.info("Ollama structured OK (attempt 1: json mode)")
                return result
            else:
                logger.warning("Attempt 1 returned empty")
        except Exception as e:
            logger.warning("Attempt 1 failed: %s", e)

        # ===== ATTEMPT 2: JSON mode + shorter prompt =====
        try:
            short_prompt = prompt[:2500] if len(prompt) > 2500 else prompt
            json_system2 = f"Return a JSON object with fields: {fields_str}\n\nExample:\n{example}"
            content = await self._call_ollama(
                messages=[
                    {"role": "system", "content": json_system2},
                    {"role": "user", "content": short_prompt},
                ],
                temperature=0.05,
                max_tokens=max_tokens or 8192,
                force_json=True,
            )
            last_content = content if content.strip() else last_content
            if content.strip():
                json_str = self._extract_json(content)
                data = json.loads(json_str)
                data = self._fix_flat_arrays(data, schema)
                result = response_model.model_validate(data)
                logger.info("Ollama structured OK (attempt 2: json mode short)")
                return result
            else:
                logger.warning("Attempt 2 returned empty")
        except Exception as e:
            logger.warning("Attempt 2 failed: %s", e)

        # ===== ATTEMPT 3: Normal mode (no format=json) + minimal prompt =====
        try:
            mini_prompt = prompt[:1500] if len(prompt) > 1500 else prompt
            mini_system = (
                "You must respond with ONLY a JSON object. "
                "No other text. Start with { end with }.\n\n"
                f"Example:\n{example}"
            )
            content = await self._call_ollama(
                messages=[
                    {"role": "system", "content": mini_system},
                    {"role": "user", "content": mini_prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens or 8192,
                force_json=False,
            )
            last_content = content if content.strip() else last_content
            if content.strip():
                json_str = self._extract_json(content)
                data = json.loads(json_str)
                data = self._fix_flat_arrays(data, schema)
                result = response_model.model_validate(data)
                logger.info("Ollama structured OK (attempt 3: normal mode)")
                return result
            else:
                logger.warning("Attempt 3 returned empty")
        except Exception as e:
            logger.warning("Attempt 3 failed: %s", e)

        # ===== ATTEMPT 4: Ask for plain text, parse it ourselves =====
        try:
            text_system = "Answer the question. Be specific and factual."
            text_prompt = prompt[:2000] if len(prompt) > 2000 else prompt
            content = await self._call_ollama(
                messages=[
                    {"role": "system", "content": text_system},
                    {"role": "user", "content": text_prompt},
                ],
                temperature=0.3,
                max_tokens=max_tokens or 8192,
                force_json=False,
            )
            last_content = content if content.strip() else last_content

            if content.strip():
                # First try JSON extraction anyway
                try:
                    json_str = self._extract_json(content)
                    if json_str and json_str != "{}":
                        data = json.loads(json_str)
                        data = self._fix_flat_arrays(data, schema)
                        result = response_model.model_validate(data)
                        logger.info("Ollama structured OK (attempt 4: found json in text)")
                        return result
                except Exception:
                    pass

                # Parse plain text into dict
                data = self._parse_text_to_dict(content, schema)
                data = self._fix_flat_arrays(data, schema)
                result = response_model.model_validate(data)
                logger.info("Ollama structured OK (attempt 4: parsed from text)")
                return result
            else:
                logger.warning("Attempt 4 returned empty")
        except Exception as e:
            logger.warning("Attempt 4 failed: %s", e)

        # ===== FALLBACK: Build from defaults =====
        logger.error("All 4 attempts failed — building fallback from defaults")
        return self._build_fallback(response_model, last_content)

    def _build_example(self, model: Type[T]) -> str:
        """Build concrete JSON example."""
        schema = model.model_json_schema()
        props = schema.get("properties", {})
        defs = schema.get("$defs", {})
        example = {}

        for name, info in props.items():
            ftype = info.get("type", "")
            default = info.get("default", None)

            if "anyOf" in info:
                for opt in info["anyOf"]:
                    if opt.get("type"):
                        ftype = opt["type"]
                        break
                if not ftype:
                    ftype = "string"

            if default is not None:
                example[name] = default
            elif ftype == "string":
                example[name] = "example text here"
            elif ftype == "integer":
                example[name] = 0
            elif ftype == "boolean":
                example[name] = False
            elif ftype == "array":
                items = info.get("items", {})
                if "$ref" in items:
                    ref_name = items["$ref"].split("/")[-1]
                    items = defs.get(ref_name, items)
                if items.get("properties"):
                    nested = {}
                    for nk, nv in list(items["properties"].items())[:5]:
                        nt = nv.get("type", "string")
                        nested[nk] = "example" if nt == "string" else ([] if nt == "array" else 0)
                    example[name] = [nested]
                else:
                    example[name] = ["example"]
            elif ftype == "object":
                nested = {}
                for nk, nv in list(info.get("properties", {}).items())[:5]:
                    nt = nv.get("type", "string")
                    nested[nk] = "example" if nt == "string" else ([] if nt == "array" else 0)
                example[name] = nested if nested else {}
            else:
                example[name] = ""

        return json.dumps(example, indent=2)

    def _build_fallback(self, model: Type[T], raw: str) -> T:
        """Build minimal valid response."""
        schema = model.model_json_schema()
        props = schema.get("properties", {})

        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        clean = re.sub(r"[{}\[\]]", "", clean).strip()

        data = {}
        for name, info in props.items():
            ftype = info.get("type", "string")
            default = info.get("default", None)

            if "anyOf" in info:
                for opt in info["anyOf"]:
                    if opt.get("type"):
                        ftype = opt["type"]
                        break

            if default is not None:
                data[name] = default
            elif ftype == "string":
                if name in ("summary", "overview", "description", "component_interaction_summary"):
                    data[name] = clean[:500] if clean else "Analysis completed"
                else:
                    data[name] = ""
            elif ftype == "array":
                data[name] = []
            elif ftype == "integer":
                data[name] = 0
            elif ftype == "boolean":
                data[name] = False
            elif ftype == "object":
                data[name] = {}
            else:
                data[name] = None

        try:
            return model.model_validate(data)
        except Exception:
            return model.model_validate({})