# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""My Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from .models import EmailAction, EmailObservation, EmailState


class MyEnv(
    EnvClient[EmailAction, EmailObservation, EmailState]
):
    """
    Client for the My Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MyEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.subject)
        ...
        ...     result = client.step(EmailAction(action="mark_not_spam"))
        ...     print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MyEnv.from_docker_image("my_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(EmailAction(action="mark_spam"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: EmailAction) -> Dict:
        """
        Convert EmailAction to JSON payload for step message.

        Args:
            action: EmailAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[EmailObservation]:
        """
        Parse server response into StepResult[EmailObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with EmailObservation
        """
        obs_data = payload.get("observation", {})
        observation = EmailObservation(
            email_id=obs_data.get("email_id", ""),
            subject=obs_data.get("subject", ""),
            body=obs_data.get("body", ""),
            sender=obs_data.get("sender", ""),
            has_link=bool(obs_data.get("has_link", False)),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> EmailState:
        """
        Parse server response into EmailState object.

        Args:
            payload: JSON response from state request

        Returns:
            EmailState with episode fields and optional ground truth
        """
        return EmailState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", "spam_detection"),
            email_id=payload.get("email_id"),
            subject=payload.get("subject"),
            body=payload.get("body"),
            sender=payload.get("sender"),
            has_link=payload.get("has_link"),
            sender_reputation=payload.get("sender_reputation"),
            true_label=payload.get("true_label"),
        )
