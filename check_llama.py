import argparse
import json
import os
import re
import sys
from typing import Any, Optional

from llamafirewall import (
    AssistantMessage,
    LlamaFirewall,
    Role,
    ScannerType,
    ScanResult,
    Trace,
    UserMessage,
)
import oai_alignmentcheck_scanner  # noqa: F401 - registers OpenAI scanners
from dotenv import load_dotenv


ENV_API_KEY = "TOGETHER_API_KEY"


def _extract_text_from_content(content: Any) -> str:
    """
    Normalize the OpenAI-style content field into a plain string.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if "content" in content:
            return _extract_text_from_content(content["content"])
        return json.dumps(content, ensure_ascii=False)

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("content", ""))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join([part for part in parts if part])

    return str(content)


def _get_tool_function_name(tool_call: Any) -> Optional[str]:
    """
    Extract the function name from a tool call payload.
    """
    if not isinstance(tool_call, dict):
        return None

    func_val = tool_call.get("function")
    if isinstance(func_val, str):
        return func_val

    if isinstance(func_val, dict):
        name = func_val.get("name")
        if isinstance(name, str):
            return name

    name_val = tool_call.get("name")
    if isinstance(name_val, str):
        return name_val

    return None


def _load_trace_from_json(file_path: str) -> Trace:
    """
    Load a trace from an AgentDojo-style JSON transcript.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    messages = payload.get("messages", [])
    trace: Trace = []
    user_added = False

    for message in messages:
        role = (message.get("role") or "").lower()
        tool_calls = message.get("tool_calls")

        # Skip system messages entirely.
        if role == "system":
            continue

        # Remove assistant messages that invoke tools.
        if tool_calls:
            if role == "assistant":
                for tool_call in tool_calls:
                    func_name = _get_tool_function_name(tool_call)
                    if func_name == "post_webpage":
                        return trace
                continue
            else:
                continue

        # Tool responses are not part of the trace we want to scan.
        if role == "tool":
            continue

        text_content = _extract_text_from_content(message.get("content"))
        if not text_content.strip():
            continue

        if role == "user":
            if user_added:
                # Enforce a single user query.
                continue
            trace.append(UserMessage(content=text_content))
            user_added = True
            continue

        # Default to treating everything else as assistant output.
        agent_name = message.get("name") or message.get("role") or "assistant"
        assistant_content = json.dumps(
            {
                "agent_name": agent_name,
                "message": text_content,
            },
            ensure_ascii=False,
        )
        trace.append(AssistantMessage(content=assistant_content))

    return trace


def _load_legacy_trace(file_path: str, exec_agent: str, search_str: str) -> Trace:
    """
    Load a trace from the legacy text transcript format.
    """
    trace: Trace = []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    message_pattern = r"---------- TextMessage \(([^)]+)\) ----------"
    parts = re.split(message_pattern, content)

    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break

        agent_name = parts[i].strip()
        message_content = parts[i + 1].strip()

        if not message_content:
            continue

        if agent_name.lower() == "user":
            trace.append(UserMessage(content=message_content))
        else:
            if exec_agent in agent_name and search_str in message_content:
                break
            assistant_content = json.dumps(
                {"agent_name": agent_name, "message": message_content},
                ensure_ascii=False,
            )
            trace.append(AssistantMessage(content=assistant_content))

    return trace


def load_trace(file_path: str, exec_agent: str = "Executor", search_str: str = "TERMINATE") -> Trace:
    """
    Load a conversation trace, supporting both JSON and legacy text formats.
    """
    if file_path.lower().endswith(".json"):
        return _load_trace_from_json(file_path)

    return _load_legacy_trace(file_path, exec_agent, search_str)


def run_alignment_check(trace: Trace, model: str = "llama") -> Optional[ScanResult]:
    """
    Run AlignmentCheck scanner on the provided trace.

    Args:
        trace: The conversation trace to analyze.
        model: The model to use ("llama" for default, or OpenAI model name).

    Returns:
        Optional[ScanResult]: The scan result or None if an error occurred.
    """
    if model.lower() == "llama":
        # Use default LlamaFirewall with AlignmentCheck scanner
        scanner_config = {
            Role.ASSISTANT: [ScannerType.AGENT_ALIGNMENT, ScannerType.PROMPT_GUARD, ScannerType.CODE_SHIELD],
        }
    else:
        # Use the scanner instance in the config
        assert model in ['gpt-4o-mini', 'o4-mini', 'gpt-4o', 'gpt-5']
        scanner_config = {
            Role.ASSISTANT: [
                f'oai_alignmentcheck_{model}',  # Use the scanner instance directly
                ScannerType.PROMPT_GUARD, 
                ScannerType.CODE_SHIELD
            ]
        }
        
    firewall = LlamaFirewall(scanner_config)
    return firewall.scan_replay(trace)


def display_scan_result(result: Optional[ScanResult], description: str) -> None:
    """
    Display the scan result in a formatted way.

    Args:
        result: The scan result to display.
        description: Description of the scenario being tested.
    """
    print(f"\n\n=== Testing {description} ===")

    if result is None:
        print("Scan failed to produce a result.")
        return

    print(
        f"=== Scan Result ===\n"
        f"Score: {result.score:.4f}\n"
        f"Decision: {result.decision}\n"
        f"Reason: {result.reason}\n"
        f"=================="
    )


def check_environment(model: str = "llama") -> bool:
    """
    Check if the required environment variables are set.

    Args:
        model: The model to use ("llama" for default, or OpenAI model name).

    Returns:
        bool: True if all required environment variables are set, False otherwise.
    """
    if model.lower() == "llama":
        if not os.environ.get(ENV_API_KEY):
            print(
                f"{ENV_API_KEY} environment variable is not set. Please set it before running this demo."
            )
            print("You can get an API key from https://together.ai")
            return False
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "OPENAI_API_KEY environment variable is not set. Please set it before running with OpenAI models."
            )
            print("You can get an API key from https://openai.com")
            return False
    return True


def main() -> int:
    """
    Main function to run the demo.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run alignment check with different models")
    parser.add_argument(
        "--model", 
        choices=("llama", "gpt-4o-mini", "o4-mini", "gpt-4o", "gpt-5"),
        required=True,
        help="Model to use for alignment check"
    )
    parser.add_argument(
        "--scenario", 
        choices=("code", "cua"),
        required=True,
        help="Scenario to use for alignment check"
    )
    parser.add_argument(
        "--file", 
        type=str,
        required=True,
        help="Path to the trace file to analyze (JSON or legacy text)"
    )
    
    args = parser.parse_args()
    
    if not check_environment(args.model):
        return 1
    
    
    trace = load_trace(args.file, exec_agent="Executor" if args.scenario == 'code' else "Emailer")
    result = run_alignment_check(trace, args.model)
    
    model_description = f"AlignmentCheck (Model: {args.model})"
    display_scan_result(result, model_description)
    return 0


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())