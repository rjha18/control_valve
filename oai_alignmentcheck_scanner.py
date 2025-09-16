import logging
import os
from typing import TypeVar
from openai import AsyncOpenAI

# Assuming these imports exist - adjust based on your llamafirewall structure
from llamafirewall import register_llamafirewall_scanner
from llamafirewall.scanners import AlignmentCheckScanner 

# Set up logging
LOG = logging.getLogger(__name__)

# Type variable for output schema
OutputSchemaT = TypeVar('OutputSchemaT')


class OAIAlignmentCheckScanner(AlignmentCheckScanner):
    def __init__(self, model: str, temperature: float = None, **kwargs):
        super().__init__(**kwargs)
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = model
        if temperature is not None:  # override default temperature
            self.temperature = temperature


    async def _evaluate_with_llm(self, text: str) -> OutputSchemaT:
        try:
            # Build the messages array for OpenAI API format
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ]
            
            # Prepare chat completion arguments
            completion_args = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            # Use structured outputs with Pydantic model schema
            completion_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.output_schema.__name__,
                    "schema": self.output_schema.model_json_schema()
                }
            }
            
            response = await self.client.chat.completions.create(**completion_args)
            
            # Extract the content from OpenAI response
            content = response.choices[0].message.content
            
            LOG.info(f"[LlamaFirewall] LLM-based scanner response: {content}")
            
            # Parse JSON response and convert to Pydantic model
            import json
            try:
                parsed_content = json.loads(content)
                return self.output_schema.model_validate(parsed_content)
            except (json.JSONDecodeError, ValueError) as e:
                LOG.warning(f"Failed to parse/validate response: {e}, returning raw content")
                return content
            
        except Exception as e:
            LOG.error(f"Error in LLM evaluation: {e}")
            return self._get_default_error_response()


@register_llamafirewall_scanner("oai_alignmentcheck_gpt-4o-mini")
class OAIAlignmentCheckScanner4oMini(OAIAlignmentCheckScanner):
    def __init__(self, **kwargs):
        super().__init__(model='gpt-4o-mini', **kwargs)

@register_llamafirewall_scanner("oai_alignmentcheck_o4-mini")
class OAIAlignmentCheckScannerO4Mini(OAIAlignmentCheckScanner):
    def __init__(self, **kwargs):
        super().__init__(model='o4-mini', temperature=1.0, **kwargs)

@register_llamafirewall_scanner("oai_alignmentcheck_gpt-4o")
class OAIAlignmentCheckScanner4o(OAIAlignmentCheckScanner):
    def __init__(self, **kwargs):
        super().__init__(model='gpt-4o', **kwargs)

@register_llamafirewall_scanner("oai_alignmentcheck_gpt-5")
class OAIAlignmentCheckScanner5(OAIAlignmentCheckScanner):
    def __init__(self, **kwargs):
        super().__init__(model='gpt-5', temperature=1.0, **kwargs)
