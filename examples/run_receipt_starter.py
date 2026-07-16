"""Queue the no-model Local Run Receipts starter on local ComfyUI only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


PROMPT_PATH = Path(__file__).with_name("receipt-starter-api.json")


def request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=data, headers={"Content-Type": "application/json"} if data else {}, method="POST" if data else "GET")
    try:
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ComfyUI returned HTTP {error.code}: {detail or 'no detail'}") from error
    except URLError as error:
        raise RuntimeError("Could not reach local ComfyUI. Start it first, then retry.") from error


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Local Run Receipts no-model starter against loopback ComfyUI.")
    parser.add_argument("--port", type=int, default=8188, help="Local ComfyUI port (default: 8188).")
    parser.add_argument("--timeout", type=float, default=30, help="Seconds to wait for completion (default: 30).")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    prompt = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    if not isinstance(prompt, dict):
        raise RuntimeError("The bundled starter prompt is not a JSON object.")

    base_url = f"http://127.0.0.1:{args.port}"
    queued = request_json(f"{base_url}/prompt", {"prompt": prompt, "client_id": str(uuid4())})
    prompt_id = queued.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError(f"ComfyUI did not return a prompt_id: {queued}")
    print(f"Queued local starter: {prompt_id}")

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        history = request_json(f"{base_url}/history/{prompt_id}")
        record = history.get(prompt_id)
        status_record = record.get("status", {}) if isinstance(record, dict) else {}
        status = status_record.get("status_str") or status_record.get("status")
        if status == "success":
            receipt_items = record.get("outputs", {}).get("3", {}).get("local_run_receipts", [])
            if not receipt_items or not isinstance(receipt_items[0], dict):
                raise RuntimeError("ComfyUI finished but did not return a Local Run Receipts result.")
            receipt = receipt_items[0]
            print(f"{receipt.get('status')}: {receipt.get('receipt_path')}")
            return 0
        if status in {"error", "failed"}:
            raise RuntimeError("ComfyUI rejected the starter prompt. Inspect the local ComfyUI queue for details.")
        time.sleep(0.25)

    raise TimeoutError(f"ComfyUI did not finish within {args.timeout:g} seconds.")


if __name__ == "__main__":
    raise SystemExit(main())
