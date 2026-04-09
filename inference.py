"""Baseline inference script for all 3 email-triage tasks.

Required env vars:
- API_BASE_URL: OpenAI-compatible LLM endpoint
- MODEL_NAME: model id
- HF_TOKEN: API key/token for the LLM endpoint

Optional env vars:
- ENV_BASE_URL: environment server URL (default: http://localhost:8000)
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Literal

import requests
from openai import OpenAI

TaskName = Literal["spam_detection", "priority_triage", "phishing_risk"]

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000").rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN", "")


TASK_ACTIONS: Dict[TaskName, List[str]] = {
    "spam_detection": ["mark_spam", "mark_not_spam"],
    "priority_triage": ["mark_high_priority", "mark_normal_priority", "mark_low_priority"],
    "phishing_risk": ["mark_high_risk", "mark_medium_risk", "mark_low_risk"],
}


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _extract_action(text: str, task: TaskName) -> str:
    valid = TASK_ACTIONS[task]
    pattern = r"\b(" + "|".join(re.escape(a) for a in valid) + r")\b"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return valid[0]


def _llm_pick_action(client: OpenAI, task: TaskName, obs: Dict[str, object]) -> str:
    actions = TASK_ACTIONS[task]
    prompt = (
        f"Task: {task}\n"
        f"Subject: {obs.get('subject', '')}\n"
        f"Body: {obs.get('body', '')}\n"
        f"Sender: {obs.get('sender', '')}\n"
        f"Has link: {obs.get('has_link', False)}\n"
        f"Instructions: {obs.get('instructions', '')}\n\n"
        f"Return exactly one action from this list: {actions}\n"
    )
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a precise email-triage policy."},
            {"role": "user", "content": prompt},
        ],
    )
    content = completion.choices[0].message.content or ""
    return _extract_action(content, task)


def run_task(task: TaskName, client: OpenAI) -> None:
    rewards: List[float] = []
    step_no = 0
    success = False
    last_error = "null"
    print(f"[START] task={task} env=email_triage model={MODEL_NAME}")
    try:
        reset_resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task}, timeout=30)
        reset_resp.raise_for_status()
        reset_data = reset_resp.json()
        obs = reset_data.get("observation", {})

        action = _llm_pick_action(client, task, obs)
        step_resp = requests.post(
            f"{ENV_BASE_URL}/step",
            json={"action": {"task": task, "action": action}},
            timeout=30,
        )
        step_resp.raise_for_status()
        step_data = step_resp.json()

        reward = float(step_data.get("reward") or 0.0)
        done = bool(step_data.get("done", False))
        step_no = 1
        rewards.append(reward)
        success = done

        print(
            "[STEP] "
            f"step={step_no} action={action} reward={reward:.2f} "
            f"done={_format_bool(done)} error={last_error}"
        )
    except Exception as exc:
        last_error = str(exc).replace("\n", " ").strip() or "unknown_error"
        print(
            "[STEP] "
            f"step={step_no} action=none reward={0.00:.2f} "
            f"done={_format_bool(False)} error={last_error}"
        )
    finally:
        rewards_csv = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={_format_bool(success)} "
            f"steps={step_no} rewards={rewards_csv}"
        )


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required")
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    for task in ("spam_detection", "priority_triage", "phishing_risk"):
        run_task(task, client)


if __name__ == "__main__":
    main()
