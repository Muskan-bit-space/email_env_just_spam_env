# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic models for the email triage multi-task environment."""

from __future__ import annotations

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field


class EmailObservation(Observation):
    """What the agent sees: email content without ground-truth labels."""

    task: Literal["spam_detection", "priority_triage", "phishing_risk"] = Field(
        default="spam_detection",
        description="Active task identifier",
    )
    instructions: str = Field(
        default="",
        description="Task-specific instructions for the current step",
    )

    email_id: str = Field(default="", description="Identifier of the current email")
    subject: str = Field(default="", description="Email subject line")
    body: str = Field(default="", description="Email body text")
    sender: str = Field(default="", description="Sender address or display name")
    has_link: bool = Field(default=False, description="Whether the email contains a URL")


class EmailAction(Action):
    """Action model across all email-triage tasks."""

    task: Literal["spam_detection", "priority_triage", "phishing_risk"] = Field(
        default="spam_detection",
        description="Task this action is for",
    )

    action: Literal[
        "mark_spam",
        "mark_not_spam",
        "mark_high_priority",
        "mark_normal_priority",
        "mark_low_priority",
        "mark_high_risk",
        "mark_medium_risk",
        "mark_low_risk",
    ] = Field(
        ...,
        description="Task-specific classification action",
    )


class EmailReward(BaseModel):
    """Scalar reward wrapper (e.g. for grading)."""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(..., description="Reward or score value")


class EmailState(State):
    """Full server-visible state, including hidden labels."""

    task: Literal["spam_detection", "priority_triage", "phishing_risk"] = Field(
        default="spam_detection",
        description="Active task identifier",
    )
    email_id: Optional[str] = Field(default=None, description="Current email id")
    subject: Optional[str] = Field(default=None, description="Current subject")
    body: Optional[str] = Field(default=None, description="Current body")
    sender: Optional[str] = Field(default=None, description="Current sender")
    has_link: Optional[bool] = Field(default=None, description="Whether email has a link")
    sender_reputation: Optional[Literal["low", "high"]] = Field(
        default=None,
        description="Sender reputation tier used in the dataset",
    )
    true_label: Optional[Literal["spam", "not_spam"]] = Field(
        default=None,
        description="Ground-truth label (never exposed via EmailObservation)",
    )
    true_priority: Optional[Literal["high", "normal", "low"]] = Field(
        default=None,
        description="Ground-truth priority label (hidden from observation)",
    )
    true_risk: Optional[Literal["high", "medium", "low"]] = Field(
        default=None,
        description="Ground-truth phishing risk label (hidden from observation)",
    )
