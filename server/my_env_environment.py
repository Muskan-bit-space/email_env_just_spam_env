# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Email triage environment — Task 1: spam_detection (OpenEnv-compatible)."""

from __future__ import annotations

import random
from typing import Any, List, Literal, Optional, TypedDict
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import EmailAction, EmailObservation, EmailReward, EmailState
except ImportError:
    from models import EmailAction, EmailObservation, EmailReward, EmailState


class _EmailRecord(TypedDict):
    id: str
    subject: str
    body: str
    sender: str
    has_link: bool
    sender_reputation: Literal["low", "high"]
    label: Literal["spam", "not_spam"]


def _build_spam_dataset() -> List[_EmailRecord]:
    """Deterministic in-memory corpus (~30 emails) with fixed labels."""
    rows: List[_EmailRecord] = [
        {
            "id": "em-001",
            "subject": "CONGRATS!!! You won $5,000,000 USD lottery",
            "body": "Claim now!!! Send your bank password and SSN to claim@totally-legit.ru",
            "sender": "prizes@free-money-now.biz",
            "has_link": True,
            "sender_reputation": "low",
            "label": "spam",
        },
        {
            "id": "em-002",
            "subject": "URGENT: Your account will be suspended",
            "body": "Verify immediately: http://fake-bank-login.example/steal",
            "sender": "security@paypa1-support.net",
            "has_link": True,
            "sender_reputation": "low",
            "label": "spam",
        },
        {
            "id": "em-003",
            "subject": "Cheap meds online — no prescription",
            "body": "Buy viagra and cialis 90% off click here now!!!",
            "sender": "pharma-deals@rx-spam.co",
            "has_link": True,
            "sender_reputation": "low",
            "label": "spam",
        },
        {
            "id": "em-004",
            "subject": "You inherited $12.4M from distant relative",
            "body": "Reply with your full legal name and wire details to release funds.",
            "sender": "barrister.nigeria@legal-mail.com",
            "has_link": False,
            "sender_reputation": "low",
            "label": "spam",
        },
        {
            "id": "em-005",
            "subject": "FREE iPhone 15 — limited time",
            "body": "Survey participants only! http://track.spam.example/offer",
            "sender": "promo@giveaway-farm.io",
            "has_link": True,
            "sender_reputation": "low",
            "label": "spam",
        },
        {
            "id": "em-006",
            "subject": "Act now: crypto investment 300% daily returns",
            "body": "Guaranteed profits. Deposit BTC to wallet address in attachment.",
            "sender": "signals@moon-coin.trade",
            "has_link": False,
            "sender_reputation": "low",
            "label": "spam",
        },
        {
            "id": "em-007",
            "subject": "Your package could not be delivered",
            "body": "Reschedule delivery: http://fake-dhl.example/track?id=scam",
            "sender": "no-reply@dhl-parcel-support.xyz",
            "has_link": True,
            "sender_reputation": "low",
            "label": "spam",
        },
        {
            "id": "em-008",
            "subject": "Invoice #88421 overdue",
            "body": "Open attached macro-enabled invoice to view payment instructions.",
            "sender": "accounts@acme-invoice-copy.ru",
            "has_link": False,
            "sender_reputation": "low",
            "label": "spam",
        },
        {
            "id": "em-009",
            "subject": "Quick question about your LinkedIn",
            "body": "Saw your profile — can we connect? I have a business proposal.",
            "sender": "recruiter.jane@gmail.com",
            "has_link": False,
            "sender_reputation": "low",
            "label": "spam",
        },
        {
            "id": "em-010",
            "subject": "Meeting notes from yesterday (shared doc)",
            "body": "Please review: http://malware-drop.example/shared-notes",
            "sender": "it-helpdesk@company-updates.co",
            "has_link": True,
            "sender_reputation": "low",
            "label": "spam",
        },
        {
            "id": "em-011",
            "subject": "Re: project timeline",
            "body": "Sounds good — can you approve the small budget increase by EOD?",
            "sender": "alex.kim@acmecorp.com",
            "has_link": False,
            "sender_reputation": "high",
            "label": "not_spam",
        },
        {
            "id": "em-012",
            "subject": "Q3 roadmap review — invite",
            "body": "Calendar invite attached. Deck: https://acmecorp.com/internal/roadmap-q3",
            "sender": "pmo@acmecorp.com",
            "has_link": True,
            "sender_reputation": "high",
            "label": "not_spam",
        },
        {
            "id": "em-013",
            "subject": "Your order #44921 has shipped",
            "body": "Track your package: https://shop.example/orders/44921/track",
            "sender": "orders@shop.example",
            "has_link": True,
            "sender_reputation": "high",
            "label": "not_spam",
        },
        {
            "id": "em-014",
            "subject": "Dental appointment reminder",
            "body": "You are scheduled for Tue 10:30am. Reply CONFIRM to keep slot.",
            "sender": "frontdesk@citydental.health",
            "has_link": False,
            "sender_reputation": "high",
            "label": "not_spam",
        },
        {
            "id": "em-015",
            "subject": "Code review: PR #882",
            "body": "Left a few nits on the diff. https://github.com/acme/app/pull/882",
            "sender": "devbot@acmecorp.com",
            "has_link": True,
            "sender_reputation": "high",
            "label": "not_spam",
        },
        {
            "id": "em-016",
            "subject": "Lunch this week?",
            "body": "Are you free Thursday? There's a new place near the office.",
            "sender": "sam.rivera@acmecorp.com",
            "has_link": False,
            "sender_reputation": "high",
            "label": "not_spam",
        },
        {
            "id": "em-017",
            "subject": "Weekly team standup notes",
            "body": "Notes: https://wiki.acmecorp.com/team/standup-2026-04-01",
            "sender": "lead@acmecorp.com",
            "has_link": True,
            "sender_reputation": "high",
            "label": "not_spam",
        },
        {
            "id": "em-018",
            "subject": "Bank statement available",
            "body": "Your March statement is ready: https://secure.mybank.example/statements",
            "sender": "alerts@mybank.example",
            "has_link": True,
            "sender_reputation": "high",
            "label": "not_spam",
        },
        {
            "id": "em-019",
            "subject": "Conference registration confirmed",
            "body": "Thanks for registering. Agenda PDF: https://conf.example/agenda.pdf",
            "sender": "events@pytorch-conf.org",
            "has_link": True,
            "sender_reputation": "high",
            "label": "not_spam",
        },
        {
            "id": "em-020",
            "subject": "Mom — photos from the weekend",
            "body": "Here's the album link: https://photos.family.example/weekend-hike",
            "sender": "mom@family.example",
            "has_link": True,
            "sender_reputation": "high",
            "label": "not_spam",
        },
        {
            "id": "em-021",
            "subject": "Newsletter: Python tips",
            "body": "This week: dataclasses and typing. Read online: https://pytips.example/issue-42",
            "sender": "editor@pytips.example",
            "has_link": True,
            "sender_reputation": "high",
            "label": "not_spam",
        },
        {
            "id": "em-022",
            "subject": "HR: benefits enrollment window",
            "body": "Please review options on the HR portal before Friday COB.",
            "sender": "hr@acmecorp.com",
            "has_link": False,
            "sender_reputation": "high",
            "label": "not_spam",
        },
        {
            "id": "em-023",
            "subject": "Your subscription renews soon",
            "body": "Manage billing: https://billing.saas-tool.example/account",
            "sender": "billing@saas-tool.example",
            "has_link": True,
            "sender_reputation": "high",
            "label": "not_spam",
        },
        {
            "id": "em-024",
            "subject": "Fwd: interesting article on RL",
            "body": "Thought you'd like this read — https://arxiv.org/abs/0000.00000",
            "sender": "colleague@university.edu",
            "has_link": True,
            "sender_reputation": "high",
            "label": "not_spam",
        },
        {
            "id": "em-025",
            "subject": "Build failed on main",
            "body": "CI log: https://ci.acmecorp.com/job/8821/console",
            "sender": "ci@acmecorp.com",
            "has_link": True,
            "sender_reputation": "high",
            "label": "not_spam",
        },
        {
            "id": "em-026",
            "subject": "Exclusive webinar: grow your business",
            "body": "Hi {name}, loved your recent post. Join our free session: http://pitch.example/w",
            "sender": "growth@guru-marketing.io",
            "has_link": True,
            "sender_reputation": "low",
            "label": "spam",
        },
        {
            "id": "em-027",
            "subject": "Unusual sign-in activity",
            "body": "If this was not you, reset password: http://evil-login.example/reset",
            "sender": "support@micros0ft-account.net",
            "has_link": True,
            "sender_reputation": "low",
            "label": "spam",
        },
        {
            "id": "em-028",
            "subject": "Can I send you a quick voice note?",
            "body": "I promise it's short — just want to pitch our SEO package.",
            "sender": "outreach@agency-spam.io",
            "has_link": False,
            "sender_reputation": "low",
            "label": "spam",
        },
        {
            "id": "em-029",
            "subject": "Re: invoice",
            "body": "Please use this updated wire account (see attached PDF).",
            "sender": "cfo@acme-finance-imposter.co",
            "has_link": False,
            "sender_reputation": "low",
            "label": "spam",
        },
        {
            "id": "em-030",
            "subject": "Password expiring — verify within 24h",
            "body": "Keep access: http://sso-phish.example/verify",
            "sender": "it-security@hr-portal-impersonation.net",
            "has_link": True,
            "sender_reputation": "low",
            "label": "spam",
        },
    ]
    return rows


def action_to_predicted_label(action: EmailAction) -> Literal["spam", "not_spam"]:
    """Map environment action to a gold label space."""
    if action.action == "mark_spam":
        return "spam"
    return "not_spam"


def grade_spam_detection(
    action: EmailAction,
    true_label: Literal["spam", "not_spam"],
) -> EmailReward:
    """
    Deterministic grader: 1.0 if the action matches the true label, else 0.0.
    """
    predicted = action_to_predicted_label(action)
    score = 1.0 if predicted == true_label else 0.0
    return EmailReward(value=score)


class EmailEnv(Environment[EmailAction, EmailObservation, EmailState]):
    """Single-step spam classification episode."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._dataset: List[_EmailRecord] = _build_spam_dataset()
        self._rng: random.Random = random.Random()
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._current: Optional[_EmailRecord] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EmailObservation:
        self._reset_rubric()
        if seed is not None:
            self._rng = random.Random(seed)
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0
        self._current = self._rng.choice(self._dataset)
        rec = self._current
        return EmailObservation(
            email_id=rec["id"],
            subject=rec["subject"],
            body=rec["body"],
            sender=rec["sender"],
            has_link=rec["has_link"],
            done=False,
            reward=None,
            metadata={},
        )

    def step(
        self,
        action: EmailAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> EmailObservation:
        if self._current is None:
            raise RuntimeError("step() called before reset()")

        true_label = self._current["label"]
        predicted = action_to_predicted_label(action)
        reward = 1.0 if predicted == true_label else -0.5
        self._step_count += 1

        return EmailObservation(
            email_id="",
            subject="",
            body="",
            sender="",
            has_link=False,
            done=True,
            reward=reward,
            metadata={},
        )

    @property
    def state(self) -> EmailState:
        if self._current is None:
            return EmailState(
                episode_id=self._episode_id,
                step_count=self._step_count,
            )
        rec = self._current
        return EmailState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task="spam_detection",
            email_id=rec["id"],
            subject=rec["subject"],
            body=rec["body"],
            sender=rec["sender"],
            has_link=rec["has_link"],
            sender_reputation=rec["sender_reputation"],
            true_label=rec["label"],
        )
