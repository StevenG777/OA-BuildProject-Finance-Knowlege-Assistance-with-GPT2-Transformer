"""All external model calls are isolated here for easy mocking."""
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from .prompts import INSTRUCTIONS, build_prompt
from .schemas import GENERATION_FORMAT


load_dotenv(dotenv_path='./.env', verbose=True)


def create_client() -> OpenAI:
    """Fail early if credentials are missing; SDK reads env variable."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY before generating data.")
    return OpenAI(timeout=60.0, max_retries=2)


def generate_item(
        client: OpenAI, 
        source: dict, 
        content_type: str,
        model: str, 
        max_output_tokens: int
    ) -> tuple[dict, dict]:
    """Return parsed item and usage metadata; never silently accept truncation."""
    
    user_prompt = build_prompt(
        content_type=content_type,
        topic=source["topic"],
        source_url=source["url"],
        source_text=source["text"],
        subtopic=source.get("subtopic"),
    )

    response = client.responses.create(
        model=model,
        instructions=INSTRUCTIONS,
        input=user_prompt,
        text=GENERATION_FORMAT,
        max_output_tokens=max_output_tokens,
    )

    if response.status != "completed" or not response.output_text:
        raise RuntimeError(f"API response incomplete: {response.status}")
    
    item = json.loads(response.output_text)
    usage = response.usage
    counts = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None)
    }
    
    return item, counts
