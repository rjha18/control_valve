"""OpenAI-backed AlignmentCheck scanners for LlamaFirewall."""

import json
import logging
import os
from typing import TypeVar

from llamafirewall import register_llamafirewall_scanner
from llamafirewall.scanners import AlignmentCheckScanner
from openai import AsyncOpenAI


LOG = logging.getLogger(__name__)
OutputSchemaT = TypeVar("OutputSchemaT")


class OAIAlignmentCheckScanner(AlignmentCheckScanner):
    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        if temperature is not None:
            self.temperature = temperature

    async def _evaluate_with_llm(self, text: str) -> OutputSchemaT:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=self.temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.output_schema.__name__,
                        "schema": self.output_schema.model_json_schema(),
                    },
                },
            )
            content = response.choices[0].message.content or "{}"
            return self.output_schema.model_validate(json.loads(content))
        except Exception as exc:
            LOG.error("OpenAI alignment evaluation failed: %s", exc)
            return self._get_default_error_response()


@register_llamafirewall_scanner("oai_alignmentcheck_gpt-4o-mini")
class OAIAlignmentCheckScanner4oMini(OAIAlignmentCheckScanner):
    def __init__(self, **kwargs):
        super().__init__(model="gpt-4o-mini", **kwargs)


@register_llamafirewall_scanner("oai_alignmentcheck_o4-mini")
class OAIAlignmentCheckScannerO4Mini(OAIAlignmentCheckScanner):
    def __init__(self, **kwargs):
        super().__init__(model="o4-mini", temperature=1.0, **kwargs)


@register_llamafirewall_scanner("oai_alignmentcheck_gpt-4o")
class OAIAlignmentCheckScanner4o(OAIAlignmentCheckScanner):
    def __init__(self, **kwargs):
        super().__init__(model="gpt-4o", **kwargs)


@register_llamafirewall_scanner("oai_alignmentcheck_gpt-5")
class OAIAlignmentCheckScanner5(OAIAlignmentCheckScanner):
    def __init__(self, **kwargs):
        super().__init__(model="gpt-5", temperature=1.0, **kwargs)
