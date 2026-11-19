"""Load and validate environment-based settings and construct the selected LLM."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Validated runtime configuration; secrets are never logged."""

    provider: str
    model: str
    openai_api_key: str | None
    azure_api_key: str | None
    azure_endpoint: str | None
    azure_api_version: str
    azure_deployment: str | None
    workspace_root: Path
    checkpoint_db_path: Path
    thread_id: str
    max_tool_rounds: int
    max_revisions: int
    log_level: str
    nvidia_api_key: str | None = None
    nvidia_base_url: str | None = None
    guardrails_enabled: bool = True
    safety_api_key: str | None = None
    safety_model: str = "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
    safety_base_url: str | None = None
    max_input_chars: int = 10_000


def load_settings() -> Settings:
    """Read the project .env file, then validate required provider settings."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "azure", "nvidia"}:
        raise ValueError(
            "LLM_PROVIDER must be either 'openai', 'azure', or 'nvidia'."
        )

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in the environment or project .env file.")
    if provider == "azure":
        required = (
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT",
        )
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise ValueError(f"Missing Azure OpenAI settings: {', '.join(missing)}.")
    if provider == "nvidia":
        if not (os.getenv("NVIDIA_API_KEY") or os.getenv("API_KEY")):
            raise ValueError(
                "Set NVIDIA_API_KEY or API_KEY in the environment or project .env file."
            )

    guardrails_enabled = _read_bool("GUARDRAILS_ENABLED", default=True)
    safety_api_key = (
        os.getenv("NVIDIA_SAFETY_API_KEY")
        or os.getenv("NVIDIA_API_KEY")
        or os.getenv("API_KEY")
    )
    if guardrails_enabled and not safety_api_key:
        raise ValueError(
            "Set NVIDIA_SAFETY_API_KEY, NVIDIA_API_KEY, or API_KEY for safety checks."
        )

    max_tool_rounds = _read_bounded_int("MAX_TOOL_ROUNDS", default=4, maximum=8)
    max_revisions = _read_bounded_int("MAX_REVISIONS", default=1, maximum=3)
    max_input_chars = _read_bounded_int(
        "MAX_INPUT_CHARS", default=10_000, maximum=100_000
    )
    workspace_root = Path(
        os.getenv("WORKSPACE_ROOT", str(PROJECT_ROOT / "workspace_data"))
    ).expanduser()
    checkpoint_db_path = Path(
        os.getenv("CHECKPOINT_DB_PATH", str(PROJECT_ROOT / "data" / "checkpoints.sqlite"))
    ).expanduser()

    return Settings(
        provider=provider,
        model=(
            os.getenv("NVIDIA_MODEL")
            if provider == "nvidia"
            else os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        ) or (
            "nvidia/nemotron-3.5-lightning-30b-a3b"
            if provider == "nvidia"
            else "gpt-4.1-mini"
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        nvidia_api_key=os.getenv("NVIDIA_API_KEY") or os.getenv("API_KEY"),
        nvidia_base_url=os.getenv(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ),
        guardrails_enabled=guardrails_enabled,
        safety_api_key=safety_api_key,
        safety_model=os.getenv(
            "NVIDIA_SAFETY_MODEL",
            "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
        ),
        safety_base_url=os.getenv(
            "NVIDIA_SAFETY_BASE_URL",
            os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        ),
        max_input_chars=max_input_chars,
        workspace_root=workspace_root.resolve(),
        checkpoint_db_path=checkpoint_db_path.resolve(),
        thread_id=os.getenv("LANGGRAPH_THREAD_ID", "default"),
        max_tool_rounds=max_tool_rounds,
        max_revisions=max_revisions,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


def create_chat_model(settings: Settings) -> ChatOpenAI | AzureChatOpenAI:
    """Create a temperature-zero chat model from the validated configuration."""
    if settings.provider == "nvidia":
        if not settings.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY or API_KEY is required for the NVIDIA provider.")
        return ChatOpenAI(
            model=settings.model,
            temperature=0,
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
        )

    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider.")
        return ChatOpenAI(
            model=settings.model,
            temperature=0,
            api_key=settings.openai_api_key,
        )

    if (
        not settings.azure_api_key
        or not settings.azure_endpoint
        or not settings.azure_deployment
    ):
        raise ValueError("Azure OpenAI key, endpoint, and deployment are required.")
    return AzureChatOpenAI(
        azure_deployment=settings.azure_deployment,
        api_version=settings.azure_api_version,
        azure_endpoint=settings.azure_endpoint,
        api_key=settings.azure_api_key,
        temperature=0,
    )


def _read_bounded_int(name: str, default: int, maximum: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}.")
    return value


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")
