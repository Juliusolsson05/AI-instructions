## The AI Creativity Issue

AI agents and models are wired for "cleverness." Give them a straightforward task—like fetching data from an API, parsing a config file, or handling basic user input—and they often respond by inventing an entire custom framework, wrapper library, or "enterprise-grade" solution from scratch. 

Instead of reaching for the battle-tested tools already sitting in the standard library, they spin up new classes, decorators, context managers, custom exception hierarchies, and dependency injection layers "just in case." The result? Code that feels impressive in the moment but quietly piles on complexity, hidden dependencies, and maintenance headaches that no one asked for.

### AI's Typical "Creative" Output
```python
class ApiClientFactory:
    def __init__(self, base_url: str, auth_strategy: AuthStrategy = None):
        self.session = self._build_resilient_session()
        self.auth = auth_strategy or OAuth2Strategy()
        self.retry_policy = ExponentialBackoffPolicy()

    async def execute_request(self, endpoint: str, payload: dict = None) -> ApiResponse:
        # 200+ lines of custom retry logic, logging, circuit breakers,
        # telemetry hooks, and transformation pipelines...
        ...
```

This isn't an exaggeration. The model sees a simple HTTP call and immediately dreams up a full-featured client "for scalability."

### The Reality Most Teams Actually Need
```python
import requests
from requests.exceptions import RequestException

def fetch_api_data(url: str, params: dict = None) -> dict:
    """One clear function using the standard library. No surprises."""
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        # Log once, raise a clear error—done.
        raise RuntimeError(f"API request failed: {e}") from e
```

Same outcome. Zero new files, zero custom classes, zero tech debt.

### Why AIs Default to This Pattern
- **Training data skew**: Most public code the model learned from showcases "production-ready" architectures written by engineers trying to look forward-thinking, not the quiet, stdlib-heavy scripts that actually keep companies running.
- **Creativity bias**: Models are rewarded during training for generating novel, complete-looking solutions. Using `requests` or `json` or `os` feels too "basic" to the pattern-matching engine.
- **No long-term memory of your codebase**: The AI doesn't know your team size, your deployment constraints, or that every extra file added today will be cursed by the on-call engineer six months from now.
- **Zero cost for complexity**: Generating 300 lines of new code costs the model nothing. The human (or future agent) paying the maintenance bill is invisible during generation.

### The Real Gift: Ruthless Simplicity
Demand code that does the following instead:

- Defaults to the language's standard library first, always. Only reach for third-party packages when the stdlib genuinely cannot solve the problem.
- Treats every new class, decorator, or helper file as a liability until proven otherwise through repeated real pain.
- Keeps functions small, obvious, and import-light. If a junior engineer (or another AI agent) has to read it at 3 a.m., they should understand it in under 30 seconds.
- Documents *why* a decision was made, never *what* the code is doing—especially when choosing the boring stdlib path over a flashy custom one.
- Refactors early and often to delete custom code that turned out to be unnecessary. The best AI output is the output you can safely delete later without regret.

**Bottom line**: True creativity in software isn't inventing new wheels. It's knowing when the wheel you already have is more than enough—and having the discipline to stop there. 

When you catch an AI slipping into "creativity mode," push back immediately with one question:  
*"Can this be done with only the standard library? If yes, rewrite it that way."*

That single rule will save your codebase more pain than any architecture diagram ever could.
