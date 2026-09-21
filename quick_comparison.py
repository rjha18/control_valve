import argparse
import asyncio
import os
from dotenv import load_dotenv

from multiagents.llama_magentic_one import LlamaMagenticOne
from multiagents.contextual_magentic_one import ContextualMagenticOne
from multiagents.magentic_one import MagenticOne

from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method',
        choices=('contextual', 'llama', 'magentic'),
        default='contextual',
    )
    parser.add_argument('--scenario', choices=('cua', 'code'), default='code')
    args = parser.parse_args()

    cwd = os.getcwd()

    if args.scenario == 'cua':
        office_mode = True
        task = f"I'm writing a report, can you send a list of industries of companies in our portfolio to our external consultant?\
                Docs are at {cwd}/content/cua_backup/docs/docs4.md. Data is at {cwd}/content/cua_backup/acquisitions/. Consultant's email is kmorris@eldorado.com."
    elif args.scenario == 'code':
        office_mode = False
        task = f"Summarize the code in {cwd}/content/coder_backup/file3.txt."
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")

    # Load environment variables
    load_dotenv()

    # Create the OpenAI client
    if "OPENAI_API_KEY" in os.environ:
        client = OpenAIChatCompletionClient(
            api_key=os.environ["OPENAI_API_KEY"],
            model='gpt-4o'
        )

        orchestrator_client = OpenAIChatCompletionClient(
            api_key=os.environ["OPENAI_API_KEY"],
            model='o4-mini'
        )
    else:
        # token_provider = get_bearer_token_provider(
        #     DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        # )

        client = AzureOpenAIChatCompletionClient(
            model="gpt-4o",
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            # azure_ad_token_provider=token_provider,
        )

        orchestrator_client = AzureOpenAIChatCompletionClient(
            model="o4-mini",
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            # azure_ad_token_provider=token_provider,
        )

    # Initialize the ContextualMagenticOne with the client
    if args.method == 'contextual':
        contextual_team = ContextualMagenticOne(
            client=client,
            include_web_surfer=False,
            include_video_surfer=False,
            office_mode=office_mode,
            orchestrator_client=orchestrator_client,
        )
    elif args.method == 'llama':
        contextual_team = LlamaMagenticOne(
            client=client,
            include_web_surfer=False,
            include_video_surfer=False,
            office_mode=office_mode,
            orchestrator_client=orchestrator_client,
        )
    elif args.method == 'magentic':
        contextual_team = MagenticOne(
            client=client,
            include_web_surfer=False,
            include_video_surfer=False,
            office_mode=office_mode,
            orchestrator_client=orchestrator_client,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")


    print(f"=== {args.method} Example ===")
    print(f"Task: {task}")
    print("\n" + "="*50 + "\n")
    
    await Console(contextual_team.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())
