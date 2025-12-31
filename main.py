import os
import logging
from dataclasses import dataclass
from collections.abc import Iterable
from google.cloud import firestore
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

import google.auth
import google.auth.transport.requests
import google.oauth2.id_token

class GenerateRequest(BaseModel):
    # Model defaults to "default" so missing/empty payloads don't 422
    model: str = "default"
    # Optional legacy prompt (used only to build contents when contents missing)
    prompt: str | None = None
    # Structured contents (list of Vertex-style messages: role + parts[{text}])
    contents: list[dict] = []

class FunctionCall(BaseModel):
    name: str
    args: dict

class GenerateResponse(BaseModel):
    result: str
    function_calls: list[FunctionCall] = []

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")

_request_adapter = google.auth.transport.requests.Request()

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "europe-west1")
AI_TIMEOUT_SECONDS = int(os.environ.get("AI_TIMEOUT_SECONDS", "30"))

# Function declarations for function calling - defined as constants in the backend
FUNCTION_DECLARATIONS = [
    {
        "functionDeclarations": [
            {
                "name": "allow",
                "description": "Grants additional time to the user. Use this when the user requests a break, acknowledges they need time, or when you want to give them extra time. If the user mentions a duration (e.g., 'back in 15 minutes'), use that duration. If no duration is given, choose a reasonable default (e.g., 10-20 minutes based on the activity).",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "minutes": {
                            "type": "INTEGER",
                            "description": "The number of minutes to grant. Must be a positive integer."
                        },
                        "app": {
                            "type": "STRING",
                            "description": "Optional. The name of the specific app to allow time for. If not provided, the allowance applies to all apps."
                        }
                    },
                    "required": ["minutes"]
                }
            },
            {
                "name": "remember",
                "description": "Stores information in the AI's memory so it can be recalled in future conversations. Use this to save important facts, preferences, or context that should persist across conversations. If minutes is provided, the memory will expire after that duration. If minutes is omitted, the memory is permanent.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "content": {
                            "type": "STRING",
                            "description": "The information to remember. Should be a clear, concise fact or piece of context."
                        },
                        "minutes": {
                            "type": "INTEGER",
                            "description": "Optional. The number of minutes after which this memory should expire. If omitted, the memory is permanent."
                        }
                    },
                    "required": ["content"]
                }
            }
        ]
    }
]


@dataclass
class AuthenticatedUser:
    user_id: str
    email: str | None = None


class UserStatus:
    PENDING = "pending"
    ACTIVE = "active"
    REFUSED = "refused"


class AccessErrorCode:
    PENDING_APPROVAL = "PENDING_APPROVAL"
    ACCESS_REFUSED = "ACCESS_REFUSED"

@app.get("/ping")
def ping():
    return {"status": "ok"}

if GOOGLE_CLIENT_ID:
    db = firestore.Client()
else:
    db = None

MONTHLY_USD_LIMIT = 3.00  # for example
RESET_DAYS = 30

def _require_db():
    if db is None:
        logger.error("Firestore client unavailable")
        raise HTTPException(status_code=500, detail="Database unavailable")
    return db


def get_or_create_user(user: AuthenticatedUser):
    active_db = _require_db()
    doc_ref = active_db.collection("users").document(user.user_id)
    doc = doc_ref.get()
    if doc.exists:
        existing = doc.to_dict() or {}
        if user.email and existing.get("email") != user.email:
            existing["email"] = user.email
            existing["updatedAt"] = firestore.SERVER_TIMESTAMP
            doc_ref.set(existing)
        return existing

    # user not existing, creating in PENDING state for approval 
    user_record = {
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
        "status": UserStatus.PENDING,
        "email": user.email,
    }
    doc_ref.set(user_record)
    return user_record


def ensure_user_is_active(user_doc: dict | None):
    status = (user_doc or {}).get("status") or UserStatus.PENDING
    if status == UserStatus.PENDING:
        raise HTTPException(
            status_code=403,
            detail={
                "code": AccessErrorCode.PENDING_APPROVAL,
                "message": "User pending approval",
            },
        )
    if status == UserStatus.REFUSED:
        raise HTTPException(
            status_code=403,
            detail={
                "code": AccessErrorCode.ACCESS_REFUSED,
                "message": "User access refused",
            },
        )
    if status != UserStatus.ACTIVE:
        logger.error("Unknown user status: %s", status)
        raise HTTPException(status_code=403, detail="User status invalid")

def ensure_active_subscription(user_id: str):
    active_db = _require_db()
    sub_ref = active_db.collection("subscriptions").document(user_id)
    sub_doc = sub_ref.get()
    if not sub_doc.exists or not sub_doc.to_dict().get("createdAt"):
        raise HTTPException(status_code=402, detail=f"Subscription for {user_id} never created")
    values = sub_doc.to_dict()
    if values.get("deactivatedAt"):
        raise HTTPException(status_code=402, detail=f"Subscription for {user_id} deactivated at: {values.get('deactivatedAt')}")

def check_usage(user_id: str, cost: float, actual: bool = False):
    """
    When actual is False we only pre-check quota without writing; actual=True writes atomically.
    """
    usage_ref = db.collection("usage").document(user_id)
    now = datetime.now(timezone.utc)

    def _coerce_snapshot(doc):
        """
        Firestore's transactional get may yield a generator/iterator in some cases;
        normalize to a single snapshot (or None) so downstream logic sees a consistent shape.
        """
        if doc is None or hasattr(doc, "exists"):
            return doc
        if isinstance(doc, Iterable) and not isinstance(doc, (str, bytes)):
            try:
                return next(iter(doc))
            except StopIteration:
                return None
        return doc

    def _build_usage(doc):
        doc = _coerce_snapshot(doc)
        if doc is None or not doc.exists:
            return {
                "lastReset": now,
                "usdUsed": 0.0,
                "updatedAt": now,
            }

        usage = doc.to_dict()
        last_reset = usage.get("lastReset")
        if isinstance(last_reset, datetime):
            if now - last_reset > timedelta(days=RESET_DAYS):
                usage["lastReset"] = now
                usage["usdUsed"] = 0.0
        else:
            usage["lastReset"] = now
            usage["usdUsed"] = 0.0
        return usage

    def _validate_quota(usage_data):
        current_used = float(usage_data.get("usdUsed", 0.0))
        new_used = current_used + cost
        if new_used > MONTHLY_USD_LIMIT:
            mode = "actual" if actual else "estimate"
            raise HTTPException(status_code=429, detail=f"Monthly quota exceeded ({mode})")
        return new_used

    if not actual:
        usage_doc = usage_ref.get()
        usage = _build_usage(usage_doc)
        _validate_quota(usage)
        return

    @firestore.transactional
    def tx_logic(tx):
        doc = tx.get(usage_ref)
        usage = _build_usage(doc)
        new_used = _validate_quota(usage)
        usage["usdUsed"] = new_used
        usage["updatedAt"] = now
        tx.set(usage_ref, usage)

    tx = db.transaction()
    tx_logic(tx)

MODEL_PRICES = {
        "default": {
            "units": "M",
            "base": {
                "input": 0.50,
                "output": 1.50,
        },
    }       
}

def _tokens_per_unit(units: str) -> float:
    """
    Returns the token divisor for the pricing unit.
    """
    if units.upper() == "K":
        return 1_000.0
    if units.upper() == "M":
        return 1_000_000.0
    logger.error(f"Invalid units: {units}")
    raise HTTPException(status_code=500, detail=f"Invalid data for model")


def _calculate_tiered_cost(
    tokens: int,
    base_rate: float,
    extra_rate: float,
    threshold: float | None,
    tokens_per_unit: float,
) -> float:

    if threshold is None or tokens <= threshold:
        return (tokens / tokens_per_unit) * base_rate

    # yes, if we exceed the threshold, the extra rate is charged on ALL tokens, not just the extra ones
    return (tokens / tokens_per_unit) * extra_rate


def get_model_pricing(model: str) -> dict:
    pricing = MODEL_PRICES.get(model)

    if pricing is not None:
        return pricing

    if db is None:
        logger.warning("Model pricing not found and Firestore unavailable; using defaults")
        return MODEL_PRICES["default"]

    try:
        doc = db.collection("model_prices").document(model).get()
        if not doc.exists:
            logger.warning(f"Model pricing not found for {model}, using defaults")
            return MODEL_PRICES["default"]

        # SUCESS
        MODEL_PRICES[model] = doc.to_dict() # cache it
        return MODEL_PRICES[model]
    
    except Exception as e:
        logger.warning(f"Error loading model pricing for {model}: {e}. Using defaults")
        return MODEL_PRICES["default"] 


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = get_model_pricing(model)

    base = pricing.get("base") or {}
    extra = pricing.get("extra") or {}
    threshold = pricing.get("threshold")

    tokens_per_unit = _tokens_per_unit(pricing.get("units"))

    base_input_rate = base.get("input")
    base_output_rate = base.get("output")
    extra_input_rate = extra.get("input", base_input_rate)
    extra_output_rate = extra.get("output", base_output_rate)

    if base_input_rate is None or base_output_rate is None:
        raise HTTPException(status_code=500, detail=f"Unknown model: {model}")

    input_cost = _calculate_tiered_cost(
        input_tokens, base_input_rate, extra_input_rate, threshold, tokens_per_unit
    )
    output_cost = _calculate_tiered_cost(
        output_tokens, base_output_rate, extra_output_rate, threshold, tokens_per_unit
    )
    return input_cost + output_cost

def verify_id_token_and_get_user(
    authorization: str = Header(None),
) -> AuthenticatedUser:
    """
    Extracts the Bearer token, verifies it with Google,
    and returns the google ID.
    """

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = authorization.split(" ", 1)[1]

    try:
        idinfo = google.oauth2.id_token.verify_oauth2_token(
            token,
            _request_adapter,
            GOOGLE_CLIENT_ID,
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid ID token")

    # 'sub' is the stable user identifier
    user_sub = idinfo["sub"]
    user_email = idinfo.get("email")
    logger.info(f"Authenticated user sub={user_sub}")
    return AuthenticatedUser(user_id=user_sub, email=user_email)

def estimate_tokens(prompt: str) -> int:
    return len(prompt.split()) * 2


def _extract_usage_tokens(usage_metadata: dict) -> tuple[int | None, int | None]:
    """
    Vertex AI documents:
      promptTokenCount      -> input tokens
      candidatesTokenCount  -> output tokens (aggregated)
      totalTokenCount       -> sum (not used for billing breakdown here)
    Some client libraries expose alternative snake_case keys; we tolerate both.
    """

    def coerce_int(value) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    input_tokens = None
    for key in ("promptTokenCount", "inputTokenCount", "input_tokens"):
        input_tokens = coerce_int(usage_metadata.get(key))
        if input_tokens is not None:
            break

    output_tokens = None
    for key in ("candidatesTokenCount", "outputTokenCount", "completionTokenCount", "output_tokens"):
        output_tokens = coerce_int(usage_metadata.get(key))
        if output_tokens is not None:
            break

    return input_tokens, output_tokens


def _estimate_tokens_in(contents: list[dict]) -> int:
    result = 0
    for content in contents:
        if not isinstance(content, dict):
            continue
        parts = content.get("parts") or []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                # Rough heuristic: 2 tokens per word to stay conservative
                result += len(part["text"].split()) * 2
    return result


def _extract_text_from_response(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if candidates:
        first_candidate = candidates[0] or {}
        content = first_candidate.get("content", {}) if isinstance(first_candidate, dict) else {}
        parts = content.get("parts") or []
        text_parts = [
            part.get("text")
            for part in parts
            if isinstance(part, dict) and part.get("text")
        ]
        if text_parts:
            return "".join(text_parts)
        if isinstance(first_candidate.get("outputText"), str):
            return first_candidate["outputText"]
    if isinstance(payload.get("text"), str):
        return payload["text"]
    logger.error("AI response missing text: %s", payload)
    raise HTTPException(status_code=502, detail="AI response missing text")


def _build_model_path(model: str, project_id: str, location: str) -> str:
    if model.startswith("projects/"):
        return model
    return f"projects/{project_id}/locations/{location}/publishers/google/models/{model}"


def _extract_function_calls(payload: dict) -> list[FunctionCall]:
    """Extract function calls from Vertex AI response."""
    function_calls = []
    candidates = payload.get("candidates") or []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content", {})
        if not isinstance(content, dict):
            continue
        parts = content.get("parts", [])
        for part in parts:
            if not isinstance(part, dict):
                continue
            function_call = part.get("functionCall")
            if function_call and isinstance(function_call, dict):
                name = function_call.get("name")
                args = function_call.get("args", {})
                if name:
                    function_calls.append(FunctionCall(name=name, args=args))
    return function_calls

def call_vertex_gemini(contents: list[dict], model: str) -> tuple[str, list[FunctionCall]]:
    try:
        credentials, project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    except Exception as exc:
        logger.error("Unable to load default credentials: %s", exc)
        raise HTTPException(status_code=500, detail="AI credentials unavailable")

    project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if not project_id:
        logger.error("Project ID missing for Vertex AI call")
        raise HTTPException(status_code=500, detail="AI configuration incomplete")

    model_path = _build_model_path(model, project_id, VERTEX_LOCATION)
    url = f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/{model_path}:generateContent"

    session = google.auth.transport.requests.AuthorizedSession(credentials)
    payload = {
        "contents": contents,
        "tools": FUNCTION_DECLARATIONS  # Always include function declarations
    }
    response = session.post(url, json=payload, timeout=AI_TIMEOUT_SECONDS)

    if response.status_code >= 300:
        logger.error(
            "Vertex AI call failed status=%s body=%s", response.status_code, response.text
        )
        raise HTTPException(status_code=502, detail="AI generation failed")

    try:
        data = response.json()
    except Exception as exc:
        logger.error("Failed to parse AI response JSON: %s", exc)
        raise HTTPException(status_code=502, detail="AI response invalid")
    text = _extract_text_from_response(data)
    function_calls = _extract_function_calls(data)
    usage_meta = data.get("usageMetadata") or {}
    input_tokens, output_tokens = _extract_usage_tokens(usage_meta)

    return (text, function_calls)


# Default LLM generator can be overridden in tests
llm_generate = call_vertex_gemini

@app.post("/api/generate", response_model=GenerateResponse)
def generate(
    req: GenerateRequest,
    user: AuthenticatedUser = Depends(verify_id_token_and_get_user),  # custom type
):

    # Ensure user document exists
    user_doc = get_or_create_user(user)

    # Ensure user is approved
    ensure_user_is_active(user_doc)

    # Ensure subscription is active
    ensure_active_subscription(user.user_id)

    # Build contents payload (prefer provided structured history)
    contents = req.contents or []
    if not contents:
        if req.prompt:
            contents = [{"role": "user", "parts": [{"text": req.prompt}]}]
        else:
            raise HTTPException(status_code=400, detail="Missing contents and prompt")

    # Estimate max cost for pre-check (rough)
    estimate_tokens_in = _estimate_tokens_in(contents)
    estimate_tokens_out = 64
    est_cost = calculate_cost(req.model, estimate_tokens_in, estimate_tokens_out)

    # Pre-check quota
    check_usage(user.user_id, cost=est_cost, actual=False)

    # Call Gemini (real)
    llm_result = llm_generate(contents, req.model)
    input_tokens = llm_result.input_tokens or estimate_tokens_in
    output_tokens = llm_result.output_tokens or estimate_tokens(llm_result.text)
    actual_cost = calculate_cost(req.model, input_tokens, output_tokens)

    # Commit real cost
    check_usage(user.user_id, cost=actual_cost, actual=True)

    return GenerateResponse(result=llm_result.text)


@app.get("/api/auth/status")
def auth_status(user: AuthenticatedUser = Depends(verify_id_token_and_get_user)):
    """
    Lightweight status check invoked right after login so the client can fail fast
    (e.g., pending approval or refused access) instead of discovering it during messaging.
    """
    user_doc = get_or_create_user(user)
    ensure_user_is_active(user_doc)
    return {"status": "active"}
