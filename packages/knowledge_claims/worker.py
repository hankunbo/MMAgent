"""Isolated bounded HTTP request; secrets are resolved only in this process."""

import json
import os
import sys
import urllib.request

from .ports import LLMConfig


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    try:
        raw = sys.stdin.buffer.read(262145)
        if len(raw) > 262144:
            return 1
        envelope = json.loads(raw)
        config = LLMConfig.model_validate(envelope["config"])
        key = os.environ[config.api_key_env]
        if not key or any(ord(c) < 33 or ord(c) > 126 for c in key):
            return 1
        request = envelope["request"]
        payload = {
            "model": config.model,
            "store": False,
            "max_completion_tokens": envelope["max_output_tokens"],
            "messages": [
                {"role": "system", "content": request["prompt"]},
                {"role": "user", "content": json.dumps(request["evidence"], ensure_ascii=False)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "claim_candidates",
                    "strict": True,
                    "schema": request["schema"],
                },
            },
        }
        req = urllib.request.Request(
            config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        )
        with urllib.request.build_opener(NoRedirect()).open(
            req, timeout=config.timeout_seconds
        ) as response:
            body = response.read(65537)
        if len(body) > 65536:
            return 1
        result = json.loads(body)["choices"][0]
        if result.get("finish_reason") != "stop" or result["message"].get("refusal"):
            return 1
        content = result["message"]["content"]
        if type(content) is not str or key in content:
            return 1
        sys.stdout.write(content)
        return 0
    except Exception:
        # Never print HTTP exceptions, provider bodies, headers or credentials.
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
