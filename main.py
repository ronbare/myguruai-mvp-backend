import os
import base64
import json
import re
import time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

# --- Config via environment variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "")  # for Azure-style deployments
MOCK = os.getenv("MOCK", "0") == "1"

app = FastAPI(title="MyGuruAI-MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SolveResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    steps: Optional[str] = None
    graph_svg: Optional[str] = None
    error: Optional[str] = None

def _clean(s: str) -> str:
    s = s.replace("\\\\", "\\").replace("\r", "")
    s = re.sub(r"^```(?:json|JSON)?\s*", "", s.strip())
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _fallback_extract(text: str) -> tuple[Optional[str], str]:
    t = _clean(text)
    m = re.search(r"final\s*answer\s*[:\-]\s*(.+)$", t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        ans = m.group(1).strip()
        steps = re.split(r"final\s*answer\s*[:\-]", t, flags=re.IGNORECASE)[0].strip()
        return ans, steps
    m = re.search(r"\\boxed\{([^}]+)\}", t)
    if m:
        return m.group(1).strip(), t
    m = re.findall(r"([A-Za-z0-9_]+)\s*=\s*([^\n]+)", t)
    if m:
        var, val = m[-1]
        return f"{var} = {val}".strip(), t
    return None, t

def call_openai_with_image_base64(b64_image: str, prompt: str, language: str, need_graph: bool) -> dict:
    """
    Ask the model to return STRICT JSON:
      { "steps": "...", "final_answer": "...", [optional "graph_svg": "<svg>...</svg>"] }
    Language is enforced (English or Bahasa Melayu). Graph is included only if helpful and requested.
    """
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    params = {}
    if OPENAI_API_VERSION:
        params["api-version"] = OPENAI_API_VERSION

    graph_clause = (
        ' If a simple graph would meaningfully aid the solution AND the user requested it, '
        'include an additional key "graph_svg" containing a compact inline SVG (<=800x600), '
        'with labeled axes and no external CSS or scripts.'
        if need_graph else
        " Do not include any graph output."
    )

    system_msg = {
        "role": "system",
        "content": (
            "You are a precise math tutor. Examine the image and produce a concise, correct solution. "
            "Return STRICT JSON with exactly these keys: "
            '{"steps": "<multi-line explanation>", "final_answer": "<short final answer>"}.'
            + graph_clause +
            f" Use the user's requested language: {language}. "
            "If the language is 'Bahasa Melayu', respond in formal Bahasa Melayu with correct mathematical terminology. "
            "Absolutely no markdown, no backticks, and no extra commentary outside the JSON."
        ),
    }

    user_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
    ]

    data = {
        "model": OPENAI_MODEL,
        "messages": [
            system_msg,
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
    }

    resp = requests.post(url, headers=headers, params=params, json=data, timeout=90)
    resp.raise_for_status()
    return resp.json()

@app.post("/solve", response_model=SolveResponse)
async def solve(
    image: UploadFile = File(...),
    guidance: str = Form("Solve step-by-step and give a short final answer."),
    language: str = Form("English"),          # NEW: language toggle (English / Bahasa Melayu)
    need_graph: bool = Form(False),           # NEW: graph option (include only if applicable)
):
    # Mock mode for wiring tests
    if MOCK:
        demo_steps = (
            "1) Kenal pasti persamaan: 2x + 5 = 17\n"
            "2) Tolak 5 pada kedua-dua belah: 2x = 12\n"
            "3) Bahagi 2: x = 6"
            if language.lower().startswith("bahasa")
            else
            "1) Identify the equation: 2x + 5 = 17\n"
            "2) Subtract 5 from both sides: 2x = 12\n"
            "3) Divide by 2: x = 6"
        )
        return SolveResponse(success=True, steps=demo_steps, answer="x = 6", graph_svg=None)

    if not OPENAI_API_KEY:
        return SolveResponse(success=False, error="OPENAI_API_KEY is not set on the server.")

    # Read image and convert to base64
    content = await image.read()
    b64_image = base64.b64encode(content).decode("utf-8")

    try:
        t0 = time.time()
        result = call_openai_with_image_base64(b64_image, guidance, language, need_graph)
        # Extract assistant text
        text_out = ""
        choices = result.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            text_out = msg.get("content", "") or ""
        if not text_out:
            return SolveResponse(success=False, error="No text found in model response.")

        cleaned = _clean(text_out)
        steps = None
        answer = None
        graph_svg = None

        # Parse strict JSON first
        try:
            data = json.loads(cleaned)
            steps = _clean(data.get("steps", "") or "")
            answer = _clean(data.get("final_answer", "") or "")
            graph_svg = data.get("graph_svg") if isinstance(data, dict) else None
            if steps == "": steps = None
            if answer == "": answer = None
        except Exception:
            # Fallback to heuristic extraction
            answer, steps_text = _fallback_extract(cleaned)
            steps = steps or steps_text

        # Optional: if model returned a graph that isn't needed, ignore it
        if not need_graph:
            graph_svg = None

        return SolveResponse(success=True, answer=answer, steps=steps, graph_svg=graph_svg)

    except requests.HTTPError as e:
        try:
            detail = e.response.json()
            err_msg = detail.get("error", {}).get("message") if isinstance(detail, dict) else str(detail)
        except Exception:
            err_msg = str(e)
        return SolveResponse(success=False, error=f"OpenAI API error: {err_msg}")
    except Exception as e:
        return SolveResponse(success=False, error=str(e))
