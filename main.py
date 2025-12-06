import os
from google.cloud import firestore
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

import google.auth.transport.requests
import google.oauth2.id_token

class GenerateRequest(BaseModel):
    prompt: str
    model: str

class GenerateResponse(BaseModel):
    result: str

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")

_request_adapter = google.auth.transport.requests.Request()

app = FastAPI()

@app.get("/ping")
def ping():
    return {"status": "ok"}

if GOOGLE_CLIENT_ID:
    db = firestore.Client()
else:
    db = None

MONTHLY_USD_LIMIT = 3.00  # for example
RESET_DAYS = 30

def get_or_create_user(user_id: str):
    doc_ref = db.collection("users").document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    doc_ref.set({
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })
    return None

def ensure_active_subscription(user_id: str):
    sub_ref = db.collection("subscriptions").document(user_id)
    sub_doc = sub_ref.get()
    if not sub_doc.exists or sub_doc.to_dict().get("status") != "active":
        raise HTTPException(status_code=402, detail="No active subscription")
    return sub_doc.to_dict()

def check_and_update_usage(user_id: str, estimate_usd: float = 0.0, actual_usd: float | None = None):
    usage_ref = db.collection("usage").document(user_id)

    @firestore.transactional
    def tx_logic(tx):
        doc = tx.get(usage_ref)
        now = datetime.now(timezone.utc)

        if not doc.exists:
            usage = {
                "lastReset": now,
                "usdUsed": 0.0,
                "updatedAt": now,
            }
        else:
            usage = doc.to_dict()
            last_reset = usage["lastReset"]
            if isinstance(last_reset, datetime):
                if now - last_reset > timedelta(days=RESET_DAYS):
                    # Reset period
                    usage["lastReset"] = now
                    usage["usdUsed"] = 0.0
            else:
                # weird, just reset
                usage["lastReset"] = now
                usage["usdUsed"] = 0.0

        usd_used = float(usage.get("usdUsed", 0.0))

        # Pre-check
        if usd_used + estimate_usd > MONTHLY_USD_LIMIT:
            raise HTTPException(status_code=429, detail="Monthly quota exceeded (estimate)")

        if actual_usd is not None:
            usd_used += actual_usd
            if usd_used > MONTHLY_USD_LIMIT:
                raise HTTPException(status_code=429, detail="Monthly quota exceeded")
            usage["usdUsed"] = usd_used
            usage["updatedAt"] = now
            tx.set(usage_ref, usage)

    tx = db.transaction()
    tx_logic(tx)

MODEL_PRICES = {
    "gemini-2.0-flash": {
        "input_per_1k_tokens": 0.10,
        "output_per_1k_tokens": 0.30,
    },
    "gemini-2.0-pro": {
        "input_per_1k_tokens": 0.50,
        "output_per_1k_tokens": 1.50,
    },
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = MODEL_PRICES[model]
    input_cost = (input_tokens / 1000.0) * p["input_per_1k_tokens"]
    output_cost = (output_tokens / 1000.0) * p["output_per_1k_tokens"]
    return input_cost + output_cost

def verify_id_token_and_get_user(
    authorization: str = Header(None),
) -> str:
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
    return idinfo["sub"]

@app.post("/api/generate", response_model=GenerateResponse)
def generate(
    req: GenerateRequest,
    user_id: str = Depends(verify_id_token_and_get_user),  # custom type
):

    # Ensure user document exists
    get_or_create_user(user_id)

    # Ensure subscription is active
    ensure_active_subscription(user_id)

    # Estimate max cost for pre-check (rough)
    estimate_tokens_in = len(req.prompt.split()) * 3
    estimate_tokens_out = 512
    est_cost = calculate_cost(req.model, estimate_tokens_in, estimate_tokens_out)

    # Pre-check quota
    check_and_update_usage(user_id, estimate_usd=est_cost, actual_usd=None)

    # Call Gemini (real)
    # result, input_tokens, output_tokens = call_gemini(...)
    # Stub:
    result = f"Fake answer: {req.prompt}"
    input_tokens = estimate_tokens_in
    output_tokens = 64
    actual_cost = calculate_cost(req.model, input_tokens, output_tokens)

    # Commit real cost
    check_and_update_usage(user_id, estimate_usd=0.0, actual_usd=actual_cost)

    return GenerateResponse(result=result)
