import copy
import os
import sys
import unittest
import types
from datetime import datetime, timezone

sys.path.append(os.path.dirname(__file__))

# Minimal stubs so tests can import main.py without Firestore or Google auth deps
firestore_stub = types.SimpleNamespace(
    Client=lambda *args, **kwargs: None,
    transactional=lambda f: f,
    SERVER_TIMESTAMP="timestamp",
)
request_stub = types.SimpleNamespace(Request=lambda *args, **kwargs: None)
transport_stub = types.SimpleNamespace(requests=request_stub)
id_token_stub = types.SimpleNamespace(verify_oauth2_token=lambda *args, **kwargs: {"sub": "test-user"})

class DummyFastAPI:
    def get(self, *args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def post(self, *args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


class DummyHTTPException(Exception):
    def __init__(self, status_code: int, detail=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


fastapi_stub = types.SimpleNamespace(
    FastAPI=lambda *args, **kwargs: DummyFastAPI(),
    HTTPException=DummyHTTPException,
    Header=lambda *args, **kwargs: None,
    Depends=lambda *args, **kwargs: None,
)

google_ns = types.SimpleNamespace()
google_ns.auth = types.SimpleNamespace(transport=transport_stub)
google_ns.cloud = types.SimpleNamespace(firestore=firestore_stub)
sys.modules["google"] = google_ns
sys.modules["google.cloud"] = google_ns.cloud
sys.modules["google.cloud.firestore"] = firestore_stub
sys.modules["google.auth.transport"] = transport_stub
sys.modules["google.auth.transport.requests"] = request_stub
sys.modules["google.oauth2"] = types.SimpleNamespace(id_token=id_token_stub)
sys.modules["google.oauth2.id_token"] = id_token_stub
sys.modules["google.auth"] = google_ns.auth
sys.modules["fastapi"] = fastapi_stub

from main import MODEL_PRICES, calculate_cost, check_usage


class StubDocSnapshot:
    def __init__(self, data):
        self._data = copy.deepcopy(data) if data is not None else None

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return copy.deepcopy(self._data) if self._data is not None else None


class StubDocRef:
    def __init__(self, backing_store, user_id):
        self._store = backing_store
        self._user_id = user_id
        self.set_calls = []

    def get(self):
        return StubDocSnapshot(self._store.get(self._user_id))

    def set(self, data):
        self.set_calls.append(copy.deepcopy(data))
        self._store[self._user_id] = copy.deepcopy(data)


class StubCollection:
    def __init__(self, backing_store):
        self._store = backing_store
        self.docs = {}

    def document(self, user_id):
        if user_id not in self.docs:
            self.docs[user_id] = StubDocRef(self._store, user_id)
        return self.docs[user_id]


class StubTransaction:
    def __init__(self):
        self.get_calls = []
        self.set_calls = []

    def get(self, doc_ref):
        self.get_calls.append(doc_ref)
        return doc_ref.get()

    def set(self, doc_ref, data):
        self.set_calls.append((doc_ref, copy.deepcopy(data)))
        doc_ref.set(data)


class StubDB:
    def __init__(self, initial_usage):
        self.collections = {"usage": StubCollection(initial_usage)}
        self.transaction_calls = 0

    def collection(self, name):
        return self.collections[name]

    def transaction(self):
        self.transaction_calls += 1
        return StubTransaction()


class GeneratorTransaction(StubTransaction):
    """
    Mimics the behavior where Transaction.get can yield snapshots
    (e.g., if a query/stream were passed). Returning a generator here
    helps us detect code that assumes a direct snapshot.
    """

    def get(self, doc_ref):
        self.get_calls.append(doc_ref)

        def _gen():
            yield doc_ref.get()

        return _gen()


class GeneratorDB(StubDB):
    def transaction(self):
        self.transaction_calls += 1
        return GeneratorTransaction()


class FullStubDB(StubDB):
    """
    Extends the basic stub DB with users and subscriptions collections so that
    the generate endpoint can be exercised without Firestore.
    """

    def __init__(self, usage_store, users_store=None, subscriptions_store=None):
        super().__init__(usage_store)
        self.collections["users"] = StubCollection(users_store or {})
        self.collections["subscriptions"] = StubCollection(subscriptions_store or {})

    def collection(self, name):
        if name not in self.collections:
            self.collections[name] = StubCollection({})
        return self.collections[name]


class CalculateCostTests(unittest.TestCase):
    def setUp(self):
        self.original_prices = copy.deepcopy(MODEL_PRICES)

    def tearDown(self):
        MODEL_PRICES.clear()
        MODEL_PRICES.update(self.original_prices)

    def test_base_rates_used_when_under_threshold(self):
        MODEL_PRICES["tiered-model"] = {
            "base": {"input": 2.0, "output": 12.0},
            "extra": {"input": 4.0, "output": 18.0},
            "threshold": 200_000,
            "units": "M",
        }

        cost = calculate_cost("tiered-model", input_tokens=150_000, output_tokens=199_999)

        # 150k input @ $2 per million = 0.3; 199,999 output @ $12 per million ≈ 2.399988
        self.assertAlmostEqual(cost, 0.3 + 2.399988, places=6)

    def test_extra_rates_apply_above_threshold(self):
        MODEL_PRICES["tiered-model"] = {
            "base": {"input": 2.0, "output": 12.0},
            "extra": {"input": 4.0, "output": 18.0},
            "threshold": 200_000,
            "units": "M",
        }

        cost = calculate_cost("tiered-model", input_tokens=250_000, output_tokens=50_000)

        # Input exceeds threshold → all 250k @ $4/M = 1.0
        # Output: 50k under threshold @ $12/M = 0.6
        self.assertAlmostEqual(cost, 1.0 + 0.6, places=6)

    def test_units_k_supported(self):
        MODEL_PRICES["k-model"] = {
            "base": {"input": 0.1, "output": 0.2},
            "extra": {"input": 0.2, "output": 0.4},
            "threshold": 1_500,
            "units": "K",
        }

        cost = calculate_cost("k-model", input_tokens=2_000, output_tokens=0)

        # Input exceeds threshold → all 2,000 @ $0.2/1k = 0.4
        self.assertAlmostEqual(cost, 0.4, places=6)


class UsageAccountingTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.user_id = "existing-user"
        self.usage_store = {
            self.user_id: {
                "lastReset": self.now,
                "usdUsed": 1.0,
                "updatedAt": self.now,
            }
        }
        self.db = StubDB(self.usage_store)

        import main

        main.db = self.db

    def test_estimate_skips_transaction_and_writes(self):
        check_usage(self.user_id, cost=0.5, actual=False)

        usage_doc = self.db.collections["usage"].docs[self.user_id]
        self.assertEqual(self.db.transaction_calls, 0)
        self.assertEqual(len(usage_doc.set_calls), 0)
        self.assertEqual(self.usage_store[self.user_id]["usdUsed"], 1.0)

    def test_actual_uses_transaction_and_updates_usage(self):
        check_usage(self.user_id, cost=0.75, actual=True)

        usage_doc = self.db.collections["usage"].docs[self.user_id]
        self.assertEqual(self.db.transaction_calls, 1)
        self.assertEqual(len(usage_doc.set_calls), 1)
        self.assertAlmostEqual(self.usage_store[self.user_id]["usdUsed"], 1.75)

    def test_actual_handles_generator_transaction_get(self):
        generator_db = GeneratorDB(self.usage_store)

        import main

        main.db = generator_db

        check_usage(self.user_id, cost=0.25, actual=True)

        usage_doc = generator_db.collections["usage"].docs[self.user_id]
        self.assertEqual(generator_db.transaction_calls, 1)
        self.assertEqual(len(usage_doc.set_calls), 1)
        self.assertAlmostEqual(self.usage_store[self.user_id]["usdUsed"], 1.25)


class GenerateEndpointTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.user_id = "test-user"
        self.users_store = {
            self.user_id: {
                "createdAt": self.now,
                "updatedAt": self.now,
                "status": "active",
                "email": "tester@example.com",
            }
        }
        self.subscriptions_store = {
            self.user_id: {
                "createdAt": self.now,
                "updatedAt": self.now,
            }
        }
        self.usage_store = {
            self.user_id: {
                "lastReset": self.now,
                "usdUsed": 0.0,
                "updatedAt": self.now,
            }
        }

        import main

        main.db = FullStubDB(
            usage_store=self.usage_store,
            users_store=self.users_store,
            subscriptions_store=self.subscriptions_store,
        )

        class StubLLM:
            def __init__(self, text="Server reply", input_tokens=1200, output_tokens=800):
                self.calls = []
                self.text = text
                self.input_tokens = input_tokens
                self.output_tokens = output_tokens

            def __call__(self, contents, model):
                self.calls.append((contents, model))
                return main.LLMResult(
                    text=self.text,
                    input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens,
                )

        self.stub_llm = StubLLM()
        main.llm_generate = self.stub_llm
        self.req = main.GenerateRequest(prompt="Hello world", model="default")
        self.user_ctx = main.AuthenticatedUser(
            user_id=self.user_id, email="tester@example.com"
        )
        self.main = main

    def test_generate_uses_llm_tokens_and_updates_usage(self):
        response = self.main.generate(self.req, user=self.user_ctx)

        expected_cost = self.main.calculate_cost(
            "default", input_tokens=1200, output_tokens=800
        )
        self.assertEqual(response.result, "Server reply")
        self.assertEqual(len(self.stub_llm.calls), 1)
        contents, model = self.stub_llm.calls[0]
        self.assertEqual(contents, [{"role": "user", "parts": [{"text": "Hello world"}]}])
        self.assertEqual(model, "default")
        self.assertAlmostEqual(
            self.usage_store[self.user_id]["usdUsed"], expected_cost
        )

    def test_generate_estimates_tokens_when_usage_missing(self):
        class NullUsageLLM:
            def __init__(self):
                self.calls = []

            def __call__(self_inner, contents, model):
                self_inner.calls.append((contents, model))
                return self.main.LLMResult(
                    text="Short",
                    input_tokens=None,
                    output_tokens=None,
                )

        null_llm = NullUsageLLM()
        self.main.llm_generate = null_llm

        response = self.main.generate(self.req, user=self.user_ctx)

        expected_input = self.main.estimate_tokens("Hello world")
        expected_output = self.main.estimate_tokens("Short")
        expected_cost = self.main.calculate_cost(
            "default", expected_input, expected_output
        )

        self.assertEqual(response.result, "Short")
        self.assertAlmostEqual(
            self.usage_store[self.user_id]["usdUsed"], expected_cost
        )
        self.assertEqual(len(null_llm.calls), 1)

    def test_generate_prefers_provided_contents_over_prompt(self):
        # Build a minimal two-message history
        provided_contents = [
            {"role": "user", "parts": [{"text": "User says hi"}]},
            {"role": "model", "parts": [{"text": "AI replies"}]},
        ]

        main_module = self.main

        class CapturingLLM:
            def __init__(self):
                self.calls = []

            def __call__(self, contents, model):
                self.calls.append((contents, model))
                return main_module.LLMResult(
                    text="Done",
                    input_tokens=10,
                    output_tokens=5,
                )

        capturing_llm = CapturingLLM()
        self.main.llm_generate = capturing_llm

        req = self.main.GenerateRequest(
            prompt="legacy prompt should be ignored",
            model="default",
            contents=provided_contents,
        )

        response = self.main.generate(req, user=self.user_ctx)

        self.assertEqual(response.result, "Done")
        self.assertEqual(len(capturing_llm.calls), 1)
        contents, model = capturing_llm.calls[0]
        self.assertEqual(contents, provided_contents)
        self.assertEqual(model, "default")


class UserAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.users_store = {}
        self.subscriptions_store = {}
        self.usage_store = {}

        import main

        self.main = main
        self.db = FullStubDB(
            usage_store=self.usage_store,
            users_store=self.users_store,
            subscriptions_store=self.subscriptions_store,
        )
        self.main.db = self.db

        class CapturingLLM:
            def __init__(self):
                self.calls = []

            def __call__(self, contents, model):
                self.calls.append((contents, model))
                return self.main.LLMResult(text="ok", input_tokens=10, output_tokens=5)

        self.capturing_llm = CapturingLLM()
        self.main.llm_generate = self.capturing_llm

    def test_new_user_is_created_pending_and_returns_pending_error(self):
        user_ctx = self.main.AuthenticatedUser(
            user_id="new-user", email="newuser@example.com"
        )
        req = self.main.GenerateRequest(prompt="hi", model="default")

        with self.assertRaises(self.main.HTTPException) as ctx:
            self.main.generate(req, user=user_ctx)

        exc = ctx.exception
        self.assertEqual(exc.status_code, 403)
        self.assertEqual(
            exc.detail.get("code"), self.main.AccessErrorCode.PENDING_APPROVAL
        )
        saved_user = self.users_store["new-user"]
        self.assertEqual(saved_user["status"], self.main.UserStatus.PENDING)
        self.assertEqual(saved_user["email"], "newuser@example.com")
        self.assertEqual(len(self.capturing_llm.calls), 0)

    def test_refused_user_is_blocked(self):
        self.users_store["blocked-user"] = {
            "createdAt": self.now,
            "updatedAt": self.now,
            "status": "refused",
            "email": "blocked@example.com",
        }
        user_ctx = self.main.AuthenticatedUser(
            user_id="blocked-user", email="blocked@example.com"
        )
        req = self.main.GenerateRequest(prompt="hi", model="default")

        with self.assertRaises(self.main.HTTPException) as ctx:
            self.main.generate(req, user=user_ctx)

        exc = ctx.exception
        self.assertEqual(exc.status_code, 403)
        self.assertEqual(
            exc.detail.get("code"), self.main.AccessErrorCode.ACCESS_REFUSED
        )
        self.assertEqual(len(self.capturing_llm.calls), 0)

if __name__ == "__main__":
    unittest.main()

