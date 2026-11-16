"""DeepSeek and Claude API plans with authentic request/reply records."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import inspect
import json
import os
import time
import urllib.error
import urllib.request
from .plans import EditPlan


SYSTEM = """You are HypoLoop's scientific editing planner. Reconsider WHERE and
HOW to edit from the native inventory, returned measurements, and validation
history. Return exactly one JSON object with tensor_names, fit_spec,
justification, expected_behavior, protected_tensors, response_kind.
tensor_names contains at most eight exact parameter names or
NAME::flat[START:STOP] selectors; blocks use flattened real coordinates and
complex values interleave real and imaginary coordinates. Respect the budget.
fit_spec contains exactly loss, ridge, preservation, step, scope, condition,
weight_program. scope=full and condition=all. Regression losses are mse,
smooth_mae,huber; binary-logit losses are bce,brier. ridge is in [1e-7,1000],
preservation in [0,100], step in [0.05,1.5].
weight_program uses residual,target,prediction,age,condition with
+,-,*,/,abs,sqrt,log1p,exp,tanh,minimum,maximum,clip. Use 1 for uniform weights.
Give a concise competing explanation and a testable next-batch expectation.
response_kind is exact_affine, local_linearization, or unsupported.
Unsupported plans select no tensors. Propose a plan, never parameter values."""


@dataclass
class APIConfig:
    provider: str
    base_url: str
    model: str
    api_key: str
    style: str = "openai"
    retries: int = 5
    timeout: float = 180

    @classmethod
    def from_env(cls, provider):
        if provider not in ("deepseek", "claude"):
            raise ValueError("Choose deepseek or claude.")
        prefix = provider.upper()
        defaults = {"deepseek": ("https://api.deepseek.com", "deepseek-chat"),
                    "claude": ("", "claude-fable-5")}
        base, model = defaults[provider]
        key = os.environ.get(prefix+"_API_KEY", "")
        if not key:
            raise ValueError("Set "+prefix+"_API_KEY before using the API planner.")
        base = os.environ.get(prefix+"_BASE_URL", base)
        if not base:
            raise ValueError("Set "+prefix+"_BASE_URL to the configured service endpoint.")
        return cls(provider, base,
                   os.environ.get(prefix+"_MODEL", model), key,
                   os.environ.get(prefix+"_API_STYLE", "openai"))


class APIPlanner:
    def __init__(self, config, directory):
        self.config, self.directory = config, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.calls = 0

    def _request(self, packet):
        cfg = self.config
        user = json.dumps(packet, ensure_ascii=False)
        if cfg.style == "anthropic":
            path = "/v1/messages"
            body = dict(model=cfg.model, max_tokens=6144, system=SYSTEM,
                        messages=[dict(role="user", content=user)])
            headers = {"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01"}
        else:
            path = "/chat/completions"
            body = dict(model=cfg.model, max_tokens=6144,
                        messages=[dict(role="system", content=SYSTEM), dict(role="user", content=user)])
            headers = {"Authorization": "Bearer "+cfg.api_key}
        base = cfg.base_url.rstrip("/")
        if base.endswith(path):
            url = base
        elif cfg.style == "anthropic" and base.endswith("/v1"):
            url = base+"/messages"
        else:
            url = base+path
        headers["Content-Type"] = "application/json"
        return url, headers, body

    def propose(self, packet):
        self.calls += 1
        folder = self.directory/f"request_{self.calls:03d}"
        folder.mkdir(exist_ok=False)
        url, headers, body = self._request(packet)
        (folder/"request.json").write_text(json.dumps(body, ensure_ascii=False, indent=2))
        response = None
        for attempt in range(self.config.retries+1):
            report = dict(at=datetime.now(timezone.utc).isoformat(), attempt=attempt+1,
                          requested_model=self.config.model)
            try:
                request = urllib.request.Request(url, json.dumps(body).encode(), headers=headers)
                with urllib.request.urlopen(request, timeout=self.config.timeout) as reply:
                    raw = reply.read()
                (folder/f"reply_{attempt+1:02d}.json").write_bytes(raw)
                response = json.loads(raw)
                report["returned_model"] = response.get("model")
                report["status"] = "received"
            except (urllib.error.URLError, TimeoutError) as error:
                report.update(status="transport_error", error=str(error))
                transient = not isinstance(error, urllib.error.HTTPError) or error.code in (408,429,500,502,503,504)
                if not transient or attempt == self.config.retries:
                    (folder/f"attempt_{attempt+1:02d}.json").write_text(json.dumps(report, indent=2))
                    raise
            (folder/f"attempt_{attempt+1:02d}.json").write_text(json.dumps(report, indent=2))
            if response is not None:
                break
            time.sleep(min(2**attempt, 30))
        if self.config.style == "anthropic":
            content = "".join(item.get("text", "") for item in response["content"])
        else:
            content = response["choices"][0]["message"]["content"]
        text = content.strip()
        if text.startswith(("~~~", chr(96)*3)):
            text = "\n".join(text.splitlines()[1:-1])
        try:
            value = json.loads(text)
            plan = EditPlan.from_dict(value)
        except (ValueError, KeyError, TypeError) as error:
            (folder/"invalid_response.txt").write_text(str(error))
            raise ValueError("The recorded API response is not an executable editing plan.") from error
        (folder/"plan.json").write_text(json.dumps(plan.to_dict(), indent=2))
        return plan
