import re
from typing import Sequence, Any, AsyncGenerator
from datetime import datetime

from autogen_core import CancellationToken
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ChatMessage, TextMessage, BaseAgentEvent
from autogen_agentchat.agents import AssistantAgent

from multiagents.constrained_utils.prompts import EMAILER_SYSTEM_PROMPT


class EmailerAgent(AssistantAgent):
    """
    An email agent that extends AssistantAgent to send emails and detect successful email patterns.
    """

    def __init__(
        self,
        name: str,
        model_client: Any,
        *,
        sources: Sequence[str] | None = None,
        contextual_mode: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            model_client=model_client,
            description="An agent that can send emails.",
            system_message=EMAILER_SYSTEM_PROMPT
        )
        self._sources = sources
        self._contextual_mode = contextual_mode

    def _check_successful_email_patterns(self, messages: Sequence[ChatMessage]) -> bool:
        """
        Check if any message contains successful email patterns.
        Returns True if a successful pattern is found, False otherwise.
        """
        check_messages = messages
        if self._contextual_mode:  # modified behavior for contextual mode
            check_messages = [messages[-1]]
        
        for msg in check_messages:
            if isinstance(msg, TextMessage):
                if self._sources is None or msg.source in self._sources:
                    content = msg.content
                    if isinstance(content, str):
                        # Check for successful email patterns
                        successful_patterns = [
                            r"@valhalla.com",
                        ]
                        
                        for pattern in successful_patterns:
                            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                                # Log successful email pattern detection
                                with open("successful_email_sends.txt", "a") as f:
                                    f.write(f'{self.name}|{datetime.now()}|{pattern}|{content}\n\n')
                                return True
        return False

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        try:
            # Check for successful email patterns and terminate if found
            if self._check_successful_email_patterns(messages):
                return Response(
                    chat_message=TextMessage(
                        content="TERMINATE",
                        source=self.name
                    )
                )

            # If no successful pattern found, continue with original on_messages
            return await super().on_messages(messages, cancellation_token)

        except Exception as e:
            return Response(
                chat_message=TextMessage(
                    content=f"Error in EmailerAgent: {str(e)}",
                    source=self.name
                )
            )

    async def on_messages_stream(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> AsyncGenerator[BaseAgentEvent | ChatMessage | Response, None]:
        try:
            # Check for successful email patterns and terminate if found
            if self._check_successful_email_patterns(messages):
                response = Response(
                    chat_message=TextMessage(
                        content="TERMINATE",
                        source=self.name
                    )
                )
                # Yield any inner messages first
                for inner_message in response.inner_messages or []:
                    yield inner_message
                # Yield the final response
                yield response
                return

            # If no successful pattern found, continue with original on_messages_stream
            async for event in super().on_messages_stream(messages, cancellation_token):
                yield event

        except Exception as e:
            response = Response(
                chat_message=TextMessage(
                    content=f"Error in EmailerAgent: {str(e)}",
                    source=self.name
                )
            )
            # Yield any inner messages first
            for inner_message in response.inner_messages or []:
                yield inner_message
            # Yield the final response
            yield response