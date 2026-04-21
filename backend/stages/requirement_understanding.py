"""
Stage 2: Requirement Understanding

Extract structured information from unstructured requirement text using an LLM.
Output: requirement type, actor, optional secondary_actor, action, domain, system impact.
"""

import ast
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from stages.requirement_entry import RawRequirement
from stages.pipeline_logging import agent_log
from stages.requirement_impact_inference import enrich_impact_list


def _load_dotenv() -> None:
    """
    Load variables from a .env file in the project root (never hardcode API keys).
    Falls back silently if python-dotenv is not installed.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")


_load_dotenv()


@dataclass
class UnderstoodRequirement:
    """Structured representation of the requirement after LLM extraction."""

    type: str  # e.g. "Feature", "Product", "Sprint", "Enhancement", "Bug"
    actor: str
    action: str
    domain: str
    impact: list[str] = field(default_factory=list)
    secondary_actor: str = ""  # e.g. User when System runs the feature but humans are notified

    def to_dict(self) -> dict:
        d = {
            "type": self.type,
            "actor": self.actor,
            "action": self.action,
            "domain": self.domain,
            "impact": self.impact,
        }
        if self.secondary_actor:
            d["secondary_actor"] = self.secondary_actor
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "UnderstoodRequirement":
        if not isinstance(data, dict):
            data = {}
        impact_raw = data.get("impact", [])
        if isinstance(impact_raw, str):
            impact_raw = [s.strip() for s in impact_raw.split(",") if s.strip()] if impact_raw else []
        elif not isinstance(impact_raw, list):
            impact_raw = []
        impact = [str(x).strip() for x in impact_raw if x]
        return cls(
            type=str(data.get("type", "")).strip() or "Unknown",
            actor=str(data.get("actor", "")).strip() or "Unknown",
            action=str(data.get("action", "")).strip() or "",
            domain=str(data.get("domain", "")).strip() or "General",
            impact=impact,
            secondary_actor=str(data.get("secondary_actor", "")).strip(),
        )


EXTRACTION_PROMPT = """You are a business analyst. Extract structured information from the following product requirement.

Requirement text:
---
{requirement_text}
---

**Literal fidelity:** Preserve distinctive specifics from the text in action, domain, and impact where they carry requirements meaning (exact feature names, UI labels, themes such as "black theme", modules, systems). Prefer the source's wording over vaguer synonyms; do not generalize concrete values the user stated.

**Actor (mandatory quality):** The **actor** is the **primary** role responsible for the capability in requirements terms—**not** who builds software. **Forbidden:** phrases like "User builds system", "Team develops platform", "Developer implements".

**System-driven capabilities:** For automated or backend-led behavior (e.g. **fraud monitoring**, **fraud scoring**, **recommendation engines**, **risk alerts**, **batch/anomaly detection**, **scheduled jobs** that run without a person continuously driving them):
- Set **actor** to **"System"** (the platform/service performs the monitoring, scoring, or generation).
- Set **secondary_actor** to **"User"**, **"Admin"**, or another human role **only if** the text says they **receive notifications**, **review cases**, **configure rules**, **dismiss alerts**, or otherwise **interact** with outputs—**not** because you assume a human "does" the monitoring.
- Do **not** assign **User** as primary actor just because end users benefit from recommendations or alerts; the **System** still runs the logic unless the user is explicitly the one performing the checks.
- For normal end-user flows (search, pay, subscribe), **actor** remains **User** (or Admin, etc.) and **secondary_actor** is **""**.

**Impact (grounded but useful):** List **specific** applications, data areas, or capabilities that are **explicitly named** OR **clearly implied** by the requirement. Use **short, business-meaningful labels** (2–4 words). You do **not** need to list every possible system—the pipeline will **merge** your list with domain/action-aware hints, so prefer **high-confidence** items only; **maximum 5** in the final output after merge.

**Single product context (mandatory):** Treat the text as **one** product or initiative unless the user clearly switches to a different product. **domain**, **action**, and **impact** must all read as the **same** problem space (e.g. meal delivery → delivery, orders, meals—**not** unrelated industries mixed in). Prefer **one consolidated label** over several near-duplicates (e.g. avoid listing both "Subscription" and "Subscription billing" separately—use **one** label such as **Subscription & billing** when both mean the same billing relationship).

Extract and return ONLY a valid JSON object with exactly these keys (no other text, no markdown):
- "type": one of "Feature", "Product", "Sprint", "Enhancement", "Bug" (closest match)
- "actor": primary role—**System** for automated monitoring/recommendations/alerts as above; otherwise User, Admin, Customer, etc.
- "secondary_actor": human role that touches notifications, reviews, or config (**""** if none)
- "action": what happens (short phrase)—for System-primary, describe the automated capability (e.g. "Detect fraudulent transactions and raise alerts")
- "domain": the business domain or area (e.g. Subscription, Billing, Delivery, Settings)
- "impact": list of affected areas/capabilities per the impact rules above (may be [])

Example (user flow): {{"type": "Feature", "actor": "User", "secondary_actor": "", "action": "Pause subscription", "domain": "Subscription", "impact": ["Subscription billing", "Delivery scheduling"]}}
Example (system-driven): {{"type": "Feature", "actor": "System", "secondary_actor": "User", "action": "Monitor transactions for fraud and notify account holders of suspicious activity", "domain": "Payment", "impact": ["Billing", "Payment Gateway", "Notifications"]}}

JSON only:"""

def _parse_json_or_literal(s: str) -> dict:
    """Parse a string as JSON or, if that fails, as a Python literal (handles single quotes)."""
    s = s.strip()
    # Try standard JSON first (double quotes)
    try:
        out = json.loads(s)
        return _normalize_parsed(out)
    except json.JSONDecodeError:
        pass
    # Fallback: Grok and some LLMs return single-quoted Python-style dicts
    try:
        out = ast.literal_eval(s)
        return _normalize_parsed(out)
    except (ValueError, SyntaxError):
        pass
    raise ValueError("Could not parse as JSON or Python literal")


def _normalize_key(k: str) -> str:
    """Strip surrounding quotes/whitespace so '"type"' or \\"type\\" becomes 'type'."""
    s = str(k).strip()
    # Remove escaped quotes around the key (e.g. \\"type\\" from JSON)
    if s.startswith('\\"') and s.endswith('\\"'):
        s = s[2:-2].strip()
    if s.startswith('\\\'') and s.endswith('\\\''):
        s = s[2:-2].strip()
    for q in ('"', "'"):
        if len(s) >= 2 and s.startswith(q) and s.endswith(q):
            s = s[1:-1].strip()
    return s


def _normalize_parsed(out) -> dict:
    """Ensure we have a dict (unwrap single-element list) and normalized string keys."""
    if isinstance(out, list) and len(out) == 1:
        out = out[0]
    if not isinstance(out, dict):
        raise ValueError("Parsed value is not a JSON object or list of one object")
    return {_normalize_key(k): v for k, v in out.items()}


def _extract_json_from_response(response: str) -> dict:
    """Get JSON from LLM response, handling markdown code blocks and single-quoted output."""
    text = response.strip()
    # Try direct parse first
    try:
        return _parse_json_or_literal(text)
    except ValueError:
        pass
    # Try inside markdown code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        try:
            return _parse_json_or_literal(match.group(1).strip())
        except ValueError:
            pass
    # Try first balanced { ... } span (in case there is trailing text)
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return _parse_json_or_literal(text[start : i + 1])
                    except ValueError:
                        pass
                    break
    raise ValueError("No valid JSON found in LLM response")


def _get_llm_client_and_model() -> tuple:
    """Resolve *first available* LLM client and model from env (Groq → Gemini → xAI → OpenAI → OpenRouter)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "Step 2 requires the openai package. Install with: pip install openai"
        ) from None

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    xai_key = os.environ.get("XAI_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

    if groq_key:
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        return client, model
    if gemini_key:
        return None, os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    if xai_key:
        client = OpenAI(api_key=xai_key, base_url="https://api.x.ai/v1")
        model = os.environ.get("XAI_MODEL", "grok-4-1-fast-non-reasoning")
        return client, model
    if openai_key:
        client = OpenAI(api_key=openai_key)
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return client, model
    if openrouter_key:
        or_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
        referer = os.environ.get("OPENROUTER_HTTP_REFERER", "https://localhost").strip()
        title = os.environ.get("OPENROUTER_APP_NAME", "AI Business Analyst").strip()
        client = OpenAI(
            api_key=openrouter_key,
            base_url=or_base,
            default_headers={"HTTP-Referer": referer, "X-Title": title},
        )
        model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        return client, model
    raise RuntimeError(
        "Set API keys via a .env file in the project root (see .env.example) or environment variables:\n"
        "  GROQ_API_KEY   — for Groq (fast, free tier)\n"
        "  GEMINI_API_KEY — for Google Gemini\n"
        "  XAI_API_KEY    — for Grok (xAI)\n"
        "  OPENAI_API_KEY — for OpenAI\n"
        "  OPENROUTER_API_KEY — for OpenRouter (any supported chat model)"
    )


def get_llm_provider_and_model() -> str:
    """
    Return which LLM is **primary** and which other keys are configured (for **fallback** on rate limits).
    """
    labels: list[str] = []
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get(
        "GOOGLE_API_KEY", ""
    ).strip()
    xai_key = os.environ.get("XAI_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if groq_key:
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        labels.append(f"Groq ({model})")
    if gemini_key:
        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        labels.append(f"Gemini ({model})")
    if xai_key:
        model = os.environ.get("XAI_MODEL", "grok-4-1-fast-non-reasoning")
        labels.append(f"Grok / xAI ({model})")
    if openai_key:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        labels.append(f"OpenAI ({model})")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        labels.append(f"OpenRouter ({model})")
    if not labels:
        return "None (set GROQ_API_KEY, GEMINI_API_KEY, XAI_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY)"
    primary = labels[0]
    if len(labels) == 1:
        return primary
    fb = ", ".join(labels[1:])
    return f"{primary} — fallbacks on 429/quota: {fb}"


def _call_gemini_raw(api_key: str, model: str, prompt: str) -> str:
    """Call Google Gemini API via REST (no extra SDK)."""
    import urllib.error
    import urllib.request

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1},
    })
    req = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500]
        raise ValueError(f"Gemini API HTTP error {e.code}: {body}") from e
    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini API returned no candidates")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    if not parts:
        raise ValueError("Gemini API returned no content parts")
    text = parts[0].get("text") or ""
    return text.strip()


def _call_xai_raw(api_key: str, model: str, prompt: str) -> str:
    """Call xAI API with raw HTTP to avoid client parsing issues (e.g. KeyError on \"type\")."""
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        raise ImportError("urllib is required for xAI fallback") from None
    url = "https://api.x.ai/v1/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    })
    req = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise ValueError(f"xAI API HTTP error {e.code}: {e.read().decode()[:500]}") from e
    # Tolerate different response shapes: choices[0].message.content or quoted keys
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("xAI API returned no choices")
    msg = choices[0].get("message") if isinstance(choices[0], dict) else {}
    if not isinstance(msg, dict):
        msg = {}
    content = msg.get("content") or msg.get('"content"') or ""
    for k, v in msg.items():
        if (k.strip('"') == "content" or k == '"content"') and isinstance(v, str):
            content = v
            break
    return (content or "").strip()


def _complete_openai_chat(client, model: str, prompt: str) -> str:
    """Shared OpenAI-compatible chat completion response extraction."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    choices = getattr(response, "choices", None) or []
    first = choices[0] if choices else None
    msg = getattr(first, "message", first) if first is not None else None
    content = getattr(msg, "content", None) if msg is not None else None
    if content is None and isinstance(msg, dict):
        content = msg.get("content") or msg.get('"content"') or ""
    return (content or "").strip()


def _should_try_fallback(err: BaseException) -> bool:
    """
    True when **another** configured provider may succeed:

    - Rate limits / quota (429, 503, ``RateLimitError``, etc.).
    - Wrong or expired key for **this** provider only — try next key (401, 403, "API key not valid", …).

    Not used for generic 400 bad-request content (single-provider debugging).
    """
    try:
        from openai import APIError as _APIError
    except ImportError:
        _APIError = None
    try:
        from openai import RateLimitError as _RateLimitError
    except ImportError:
        _RateLimitError = None

    if _RateLimitError is not None and isinstance(err, _RateLimitError):
        return True
    # OpenAI SDK v1: Groq often raises APIError with ``status_code`` 429 (not always RateLimitError).
    if _APIError is not None and isinstance(err, _APIError):
        sc = getattr(err, "status_code", None)
        # 401/403: wrong key or permission for *this* provider — next key may work.
        if sc in (401, 403, 429, 503):
            return True
    code = getattr(err, "status_code", None)
    if code is None and getattr(err, "response", None) is not None:
        code = getattr(err.response, "status_code", None)
    if code in (401, 403, 429, 503):
        return True

    msg_l = str(err).lower()
    raw = str(err)
    if (
        "api key not valid" in msg_l
        or "invalid_api_key" in msg_l
        or "incorrect api key" in msg_l
        or "invalid api key" in msg_l
    ):
        return True
    if "429" in raw or "rate limit" in msg_l or "rate_limit" in msg_l or "rate_limit_exceeded" in msg_l:
        return True
    if "tokens per day" in msg_l or ("tpm" in msg_l and "exceeded" in msg_l):
        return True
    if "quota" in msg_l and ("exceed" in msg_l or "limit" in msg_l):
        return True
    # OpenAI: account out of credits / billing not set up
    if "insufficient_quota" in msg_l or "billing" in msg_l and "quota" in msg_l:
        return True
    if "too many requests" in msg_l:
        return True
    # Wrong model id (e.g. deprecated ``grok-2-latest``); next provider may work.
    if "model not found" in msg_l or ("400" in raw and "invalid argument" in msg_l):
        return True
    return False


def _configured_llm_backends() -> list[tuple[str, Callable[[str], str]]]:
    """Order: Groq → Gemini → xAI → OpenAI → OpenRouter (only backends with keys set)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "Step 2 requires the openai package. Install with: pip install openai"
        ) from None

    backends: list[tuple[str, Callable[[str], str]]] = []

    def _groq(p: str) -> str:
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY not set")
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        return _complete_openai_chat(client, model, p)

    def _gemini(p: str) -> str:
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get(
            "GOOGLE_API_KEY", ""
        ).strip()
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        return _call_gemini_raw(gemini_key, model, p)

    def _xai(p: str) -> str:
        xai_key = os.environ.get("XAI_API_KEY", "").strip()
        if not xai_key:
            raise RuntimeError("XAI_API_KEY not set")
        model = os.environ.get("XAI_MODEL", "grok-4-1-fast-non-reasoning")
        try:
            client = OpenAI(api_key=xai_key, base_url="https://api.x.ai/v1")
            return _complete_openai_chat(client, model, p)
        except (KeyError, AttributeError):
            return _call_xai_raw(xai_key, model, p)

    def _openai(p: str) -> str:
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=openai_key)
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return _complete_openai_chat(client, model, p)

    def _openrouter(p: str) -> str:
        """OpenRouter is OpenAI-compatible; model id is typically ``vendor/model`` (see openrouter.ai/models)."""
        or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not or_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        or_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
        referer = os.environ.get("OPENROUTER_HTTP_REFERER", "https://localhost").strip()
        title = os.environ.get("OPENROUTER_APP_NAME", "AI Business Analyst").strip()
        client = OpenAI(
            api_key=or_key,
            base_url=or_base,
            default_headers={"HTTP-Referer": referer, "X-Title": title},
        )
        model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        return _complete_openai_chat(client, model, p)

    if os.environ.get("GROQ_API_KEY", "").strip():
        backends.append(("Groq", _groq))
    _gk = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get(
        "GOOGLE_API_KEY", ""
    ).strip()
    if _gk:
        backends.append(("Gemini", _gemini))
    if os.environ.get("XAI_API_KEY", "").strip():
        backends.append(("Grok/xAI", _xai))
    if os.environ.get("OPENAI_API_KEY", "").strip():
        backends.append(("OpenAI", _openai))
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        backends.append(("OpenRouter", _openrouter))

    return backends


def call_llm(prompt: str) -> str:
    """
    Call an LLM. **Priority:** Groq → Gemini → xAI → OpenAI → OpenRouter (any key that is set).

    If **multiple** keys are set, ``call_llm`` **automatically tries the next provider** when the
    previous one fails with a **rate limit / quota / overload** style error (e.g. HTTP 429).

    Set ``LLM_FALLBACK=0`` to use **only** the first configured provider (no failover).

    Returns the raw response text. Used by understanding, clarification, refinement, and artifacts.
    """
    _load_dotenv()
    backends = _configured_llm_backends()
    if not backends:
        raise RuntimeError(
            "Set API keys via a .env file in the project root (see .env.example) or environment variables:\n"
            "  GROQ_API_KEY   — Groq (optional fallbacks below)\n"
            "  GEMINI_API_KEY — Google Gemini\n"
            "  XAI_API_KEY    — Grok (xAI)\n"
            "  OPENAI_API_KEY — OpenAI\n"
            "  OPENROUTER_API_KEY — OpenRouter (https://openrouter.ai/)\n"
            "Add more than one key to enable automatic fallback when Groq (or others) hit daily limits."
        )

    fb_on = os.environ.get("LLM_FALLBACK", "1").strip().lower() not in ("0", "false", "no", "off")
    if not fb_on:
        backends = backends[:1]

    last_err: BaseException | None = None
    for idx, (label, fn) in enumerate(backends):
        try:
            return fn(prompt)
        except Exception as e:
            last_err = e
            has_next = idx < len(backends) - 1
            if has_next and fb_on and _should_try_fallback(e):
                print(
                    f"[LLM] {label}: {type(e).__name__} — trying next provider...",
                    flush=True,
                )
                continue
            if not has_next and fb_on and _should_try_fallback(e):
                if len(backends) == 1:
                    raise RuntimeError(
                        "LLM rate-limited or over quota, but **no fallback key** is loaded. "
                        "Add `GEMINI_API_KEY` (and/or `OPENAI_API_KEY`, `XAI_API_KEY`) to `.env` "
                        "in the project folder, save the file, restart Streamlit, and ensure the "
                        "variable name is exactly `GEMINI_API_KEY` (or use `GOOGLE_API_KEY` for Gemini). "
                        "Set `LLM_FALLBACK=0` only if you intentionally want a single provider."
                    ) from e
                raise RuntimeError(
                    "Every configured LLM provider was tried; the **last** one also failed with a quota "
                    "or rate limit (see the original error). For **OpenAI `insufficient_quota`**: add "
                    "billing at https://platform.openai.com/account/billing , add **OPENROUTER_API_KEY** "
                    "(https://openrouter.ai/), or rely on another provider earlier in the chain "
                    "(Groq → Gemini → xAI → OpenAI → OpenRouter in `.env`)."
                ) from e
            raise
    if last_err:
        raise last_err
    raise RuntimeError("No LLM backends returned a response")


def understand_requirement(raw: RawRequirement) -> UnderstoodRequirement:
    """
    Use an LLM to extract structured fields from raw requirement text.
    Provider selection and failover match :func:`call_llm` (see env keys in ``.env.example``).
    """
    prompt = EXTRACTION_PROMPT.format(requirement_text=raw.text)
    with agent_log("Requirement Understanding"):
        content = call_llm(prompt)
        if not content:
            raise ValueError("LLM returned empty response")
        try:
            data = _extract_json_from_response(content)
            u = UnderstoodRequirement.from_dict(data)
            act = (u.actor or "").strip()
            if re.search(r"(?i)user\s+builds?\s+(the\s+)?system", act):
                u.actor = "User"
            u.impact = enrich_impact_list(
                u.action,
                u.domain,
                raw.text,
                u.impact,
                max_items=5,
            )
            return u
        except Exception as e:
            err_detail = f"{type(e).__name__}: {e!r}"
            if len(content) <= 600:
                err_detail += f"\n\nRaw LLM response:\n{content}"
            else:
                err_detail += f"\n\nRaw LLM response (first 800 chars):\n{content[:800]}..."
            raise ValueError(err_detail) from e
