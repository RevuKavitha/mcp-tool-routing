"""LLM tool selection through any OpenAI-compatible API (Groq, xAI Grok, OpenRouter, local vLLM).

- Provider is detected from the key prefix (gsk_ -> Groq, xai- -> xAI, sk-or- -> OpenRouter)
  or set explicitly.
- Every request/response is cached in a JSONL file, so an interrupted run (Colab timeout,
  rate limit) resumes without paying again. Put the cache on Google Drive in Colab.
- Tools are not executed. For multi-step tasks the model receives a simulated tool result
  and may call further tools, up to `max_turns` (the authors' Qwen-Agent runs real MCP servers).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PROVIDERS = {
    "groq": "https://api.groq.com/openai/v1",
    "xai": "https://api.x.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",      # local model, no key, no limits
    "fireworks": "https://api.fireworks.ai/inference/v1",
}

# Authors' system prompts (config.py). The released scripts use prompt_zh for the Qwen models.
PROMPT_ZH = "请判断所提供的工具是否可以用来解决用户的问题。如果可以，请选择合适的函数进行调用，无需过度思考。如果不可以，请直接回答用户的问题，无需进行过度思考。"
PROMPT_EN = ("Please determine whether the provided tools can be used to solve the user's problem. "
             "If they can, please select the appropriate function to call without overthinking. "
             "If they cannot, please directly answer the user's question without overthinking.")

SIMULATED_RESULT = json.dumps({"status": "success",
                               "result": "Simulated tool output for evaluation. Assume the call succeeded."})


def detect_provider(api_key: str) -> str:
    if api_key.startswith("gsk_"):
        return "groq"
    if api_key.startswith("xai-"):
        return "xai"
    if api_key.startswith("sk-or-"):
        return "openrouter"
    if api_key.startswith("fw_"):
        return "fireworks"
    raise ValueError("Cannot detect provider from key; pass provider= explicitly")


@dataclass
class CallResult:
    tool_calls: List[str]                      # tool names in call order (all turns)
    completion_tokens: int = 0
    prompt_tokens: int = 0
    turns: int = 0
    error: Optional[str] = None
    texts: List[str] = field(default_factory=list)


class DailyLimitError(RuntimeError):
    """The provider's daily request/token limit is used up; re-run tomorrow (the cache keeps finished work)."""


class LLMClient:
    def __init__(self, model: str, api_key: Optional[str] = None, provider: Optional[str] = None,
                 base_url: Optional[str] = None, cache_path: str = "data/llm_cache.jsonl",
                 temperature: float = 0.0, max_tokens: int = 2000, extra_body: Optional[Dict] = None,
                 min_interval: float = 0.0, max_retries: int = 8):
        from openai import OpenAI
        if provider == "ollama":
            api_key = api_key or "ollama"            # Ollama ignores the key
        api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Set LLM_API_KEY (or GROQ_API_KEY) or pass api_key=")
        if provider is None and base_url:
            provider = "custom"                   # any OpenAI-compatible endpoint given by --base-url
        provider = provider or detect_provider(api_key)
        # timeout + no hidden library retries: a stuck call fails after 60 s and our own loop retries it (visibly)
        self.client = OpenAI(api_key=api_key, base_url=base_url or PROVIDERS[provider],
                             timeout=300 if provider == "ollama" else 60, max_retries=0)
        self.model, self.provider = model, provider
        self.temperature, self.max_tokens, self.extra_body = temperature, max_tokens, extra_body or {}
        self.min_interval, self.max_retries = min_interval, max_retries
        self.cache_path, self._cache = cache_path, {}
        self._last = 0.0
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec["response"]
                    except Exception:
                        pass

    # ---------- one API call (cached, with retries) ----------
    def _chat(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        req = {"model": self.model, "messages": messages, "tools": tools, "temperature": self.temperature,
               "max_tokens": self.max_tokens, "extra_body": self.extra_body}
        key = hashlib.sha1(json.dumps(req, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        delay = 5.0
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last = time.time()
                kw = dict(model=self.model, messages=messages, temperature=self.temperature,
                          max_tokens=self.max_tokens, extra_body=self.extra_body or None)
                if tools:
                    kw.update(tools=tools, tool_choice="auto")
                r = self.client.chat.completions.create(**kw)
                msg = r.choices[0].message
                resp = {
                    "content": msg.content,
                    "tool_calls": [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                                   for tc in (msg.tool_calls or [])],
                    "completion_tokens": getattr(r.usage, "completion_tokens", 0) or 0,
                    "prompt_tokens": getattr(r.usage, "prompt_tokens", 0) or 0,
                }
                break
            except Exception as e:  # rate limit / server error / tool-call parse error
                status = getattr(e, "status_code", None)
                text = str(e)
                if status == 400 and "tool" in text.lower():
                    # e.g. Groq "tool_use_failed": model produced an invalid call -> record as no call
                    resp = {"content": None, "tool_calls": [], "completion_tokens": 0, "prompt_tokens": 0,
                            "error": text[:300]}
                    break
                if status == 413 or "too large" in text.lower():
                    raise RuntimeError(f"Request too large for this model/plan: {text[:200]}") from e
                if attempt == self.max_retries - 1:
                    raise
                low = text.lower()
                if "per day" in low or "(tpd)" in low or "(rpd)" in low:
                    raise DailyLimitError(text[:400]) from e
                retry_after = None
                try:
                    retry_after = float(e.response.headers.get("retry-after"))
                except Exception:
                    pass
                w = retry_after if retry_after else delay + random.random()
                print(f"  [API error {status}, attempt {attempt+1}/{self.max_retries}] {text[:200]} -> waiting {w:.0f}s",
                      flush=True)
                time.sleep(w)
                delay = min(delay * 2, 120)
        self._cache[key] = resp
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "response": resp}, ensure_ascii=False) + "\n")
        return resp

    # ---------- agent loop ----------
    def select_tools(self, query: str, tools: List[Dict], system_prompt: str = PROMPT_ZH,
                     max_turns: int = 1) -> CallResult:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
        out = CallResult(tool_calls=[])
        for _ in range(max_turns):
            resp = self._chat(messages, tools)
            out.turns += 1
            out.completion_tokens += resp.get("completion_tokens", 0)
            out.prompt_tokens += resp.get("prompt_tokens", 0)
            if resp.get("error"):
                out.error = resp["error"]
            if resp.get("content"):
                out.texts.append(resp["content"])
            calls = resp.get("tool_calls", [])
            if not calls:
                break
            out.tool_calls.extend(c["name"] for c in calls)
            messages.append({"role": "assistant", "content": resp.get("content") or "",
                             "tool_calls": [{"id": c["id"], "type": "function",
                                             "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                                            for c in calls]})
            for c in calls:
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": SIMULATED_RESULT})
        return out
