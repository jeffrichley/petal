"""Configuration classes for agent building."""

from typing import Annotated, Any, Dict, List, Optional, Type

from pydantic import AfterValidator, BaseModel, Field


def validate_strategy_type(v: str) -> str:
    """Validate strategy type is not empty."""
    if not v or not v.strip():
        raise ValueError("strategy_type cannot be empty")
    return v.strip()


def validate_memory_type(v: str) -> str:
    """Validate memory type is supported."""
    valid_types = ["conversation", "vector", "buffer", "summary"]
    if v not in valid_types:
        raise ValueError(f"memory_type must be one of {valid_types}")
    return v


def validate_graph_type(v: str) -> str:
    """Validate graph type is supported."""
    valid_types = ["linear", "branching", "parallel", "conditional"]
    if v not in valid_types:
        raise ValueError(f"graph_type must be one of {valid_types}")
    return v


def validate_provider(v: str) -> str:
    """Validate provider is supported."""
    valid_providers = [
        "openai",
        "anthropic",
        "google",
        "cohere",
        "huggingface",
        "ollama",
    ]
    if v not in valid_providers:
        raise ValueError(f"provider must be one of {valid_providers}")
    return v


def validate_level(v: str) -> str:
    """Validate logging level is supported."""
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if v not in valid_levels:
        raise ValueError(f"level must be one of {valid_levels}")
    return v


class StepConfig(BaseModel):
    """Configuration for a single step in an agent."""

    strategy_type: Annotated[str, AfterValidator(validate_strategy_type)] = Field(
        ..., description="Type of step strategy to use"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Configuration for the step"
    )
    node_name: Optional[str] = Field(None, description="Custom node name for the step")


class MemoryConfig(BaseModel):
    """Configuration for agent memory."""

    memory_type: Annotated[str, AfterValidator(validate_memory_type)] = Field(
        ..., description="Type of memory to use"
    )
    max_tokens: int = Field(default=500, ge=1, description="Maximum tokens to store")
    enabled: bool = Field(default=True, description="Whether memory is enabled")


class GraphConfig(BaseModel):
    """Configuration for graph building."""

    graph_type: Annotated[str, AfterValidator(validate_graph_type)] = Field(
        default="linear", description="Type of graph topology"
    )
    allow_parallel: bool = Field(default=False, description="Allow parallel execution")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")


class LLMConfig(BaseModel):
    """Configuration for LLM settings."""

    provider: Annotated[str, AfterValidator(validate_provider)] = Field(
        ..., description="LLM provider (e.g., openai, anthropic)"
    )
    model: str = Field(..., description="Model name")
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=1000, ge=1, description="Maximum tokens to generate"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging settings."""

    enabled: bool = Field(default=True, description="Whether logging is enabled")
    level: Annotated[str, AfterValidator(validate_level)] = Field(
        default="INFO", description="Logging level"
    )
    include_state: bool = Field(default=False, description="Include state in logs")


class AgentConfig(BaseModel):
    """Configuration object for agent building."""

    name: Optional[str] = Field(None, description="Name of the agent")
    state_type: Type = Field(..., description="Type for agent state")
    steps: List[StepConfig] = Field(
        default_factory=list, description="List of step configurations"
    )
    memory: Optional[MemoryConfig] = Field(None, description="Memory configuration")
    graph_config: GraphConfig = Field(
        default_factory=GraphConfig, description="Graph configuration"
    )
    llm_config: Optional[LLMConfig] = Field(None, description="LLM configuration")
    logging_config: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )

    def add_step(self, step_config: StepConfig) -> None:
        """Add a step to the configuration."""
        self.steps.append(step_config)

    def set_memory(self, memory_config: MemoryConfig) -> None:
        """Set memory configuration."""
        self.memory = memory_config

    def set_llm(self, llm_config: LLMConfig) -> None:
        """Set LLM configuration."""
        self.llm_config = llm_config

    def set_logging(self, logging_config: LoggingConfig) -> None:
        """Set logging configuration."""
        self.logging_config = logging_config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        # Note: This is a simplified version - in practice, you might want
        # to handle the state_type serialization more carefully
        config_dict = self.model_dump()
        config_dict["state_type"] = self.state_type.__name__  # Convert type to string
        return config_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create configuration from dictionary."""
        # Note: This is a simplified version - in practice, you might want
        # to handle the state_type deserialization more carefully
        if "state_type" in data and isinstance(data["state_type"], str):
            # Convert string back to type - this is simplified
            # In practice, you'd want a proper type registry
            data["state_type"] = dict  # Default to dict for now

        return cls(**data)
