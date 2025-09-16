import argparse
import json
import logging
import os
import re
import sys
from typing import Optional

from llamafirewall import (
    AssistantMessage,
    LlamaFirewall,
    Role,
    ScannerType,
    ScanResult,
    Trace,
    UserMessage,
)
from openai import AsyncOpenAI
from oai_alignmentcheck_scanner import *
from dotenv import load_dotenv


def load_trace(file_path: str, exec_agent: str = "Executor", search_str: str = "TERMINATE") -> Trace:
    trace: Trace = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content by the message delimiter pattern
    message_pattern = r'---------- TextMessage \(([^)]+)\) ----------'
    parts = re.split(message_pattern, content)
    
    # Skip the first part (before any messages) and process pairs of (agent_name, message_content)
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
            
        agent_name = parts[i].strip()
        message_content = parts[i + 1].strip()
        
        # Skip empty messages
        if not message_content:
            continue
            
        if agent_name.lower() == 'user':
            # User message - just use the content directly
            trace.append(UserMessage(content=message_content))
        else:
            # Assistant message - wrap in JSON format with agent_name and message
            if exec_agent in agent_name and search_str in message_content:
                break
            assistant_content = json.dumps({
                "agent_name": agent_name,
                "message": message_content
            }, ensure_ascii=False)
            trace.append(AssistantMessage(content=assistant_content))
    
    return trace


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
    # return firewall.scan(trace[-1], trace[:-1])
    try:
        print("Running alignment check")
    except Exception as e:
        print(f"Error running alignment check: {str(e)}")
        return None


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
        type=str, 
        required=True,
        help="Model to use for alignment check"
    )
    parser.add_argument(
        "--scenario", 
        type=str, 
        required=True,
        help="Scenario to use for alignment check"
    )
    parser.add_argument(
        "--file", 
        type=str, 
        default="/Users/rdj58/code/contextual_flow/example.txt",
        help="Path to the trace file to analyze"
    )
    
    args = parser.parse_args()
    
    assert args.scenario in ['code', 'cua'], f"Invalid scenario: {args.scenario}"
    assert args.model in ['llama','gpt-4o-mini', 'o4-mini', 'gpt-4o', 'gpt-5'], f"Invalid model: {args.model}"
    assert check_environment(args.model), f"Environment is not set for model: {args.model}"
    
    
    trace = load_trace(args.file, exec_agent="Executor" if args.scenario == 'code' else "Emailer")
    result = run_alignment_check(trace, args.model)
    
    model_description = f"AlignmentCheck (Model: {args.model})"
    display_scan_result(result, model_description)
    return 0


if __name__ == "__main__":
    load_dotenv()
    ENV_API_KEY: str = "TOGETHER_API_KEY"
    sys.exit(main())