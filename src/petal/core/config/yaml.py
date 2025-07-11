"""YAML configuration models for node loading."""

from typing import Annotated, Dict, List, Optional, Type

from pydantic import AfterValidator, BaseModel, Field, field_validator


def validate_node_type(v: str) -> str:
    """Validate node type is supported."""
    valid_types = ["llm", "react"]
    if v not in valid_types:
        raise ValueError(f"node_type must be one of {valid_types}")
    return v


def validate_provider(v: str) -> str:
    """Validate provider is supported."""
    valid_providers = ["openai", "anthropic", "google", "cohere", "huggingface"]
    if v not in valid_providers:
        raise ValueError(f"provider must be one of {valid_providers}")
    return v


def validate_temperature(v: float) -> float:
    """Validate temperature is within valid range."""
    if not 0.0 <= v <= 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0")
    return v


def validate_max_tokens(v: int) -> int:
    """Validate max_tokens is positive."""
    if v <= 0:
        raise ValueError("max_tokens must be positive")
    return v


def validate_max_iterations(v: int) -> int:
    """Validate max_iterations is positive."""
    if v <= 0:
        raise ValueError("max_iterations must be positive")
    return v


class StateSchemaConfig(BaseModel):
    """Configuration for dynamic state schema creation."""

    fields: Dict[str, Type] = Field(
        default_factory=dict, description="Field definitions for state schema"
    )
    required_fields: List[str] = Field(
        default_factory=list, description="List of required field names"
    )


class ValidationConfig(BaseModel):
    """Configuration for input/output validation."""

    input_schema: Optional[StateSchemaConfig] = Field(
        None, description="Input validation schema"
    )
    output_schema: Optional[StateSchemaConfig] = Field(
        None, description="Output validation schema"
    )


class BaseNodeConfig(BaseModel):
    """Base configuration for all node types."""

    type: Annotated[str, AfterValidator(validate_node_type)] = Field(
        ..., description="Type of node (llm, react)"
    )
    name: str = Field(..., description="Name of the node")
    description: Optional[str] = Field(None, description="Description of the node")
    enabled: bool = Field(default=True, description="Whether the node is enabled")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate node name is not empty."""
        if not v or not v.strip():
            raise ValueError("name cannot be empty")
        return v.strip()


class LLMNodeConfig(BaseNodeConfig):
    """Configuration for LLM nodes."""

    provider: Annotated[str, AfterValidator(validate_provider)] = Field(
        ..., description="LLM provider (e.g., openai, anthropic)"
    )
    model: str = Field(..., description="Model name")
    temperature: Annotated[float, AfterValidator(validate_temperature)] = Field(
        default=0.0, description="Sampling temperature"
    )
    max_tokens: Annotated[int, AfterValidator(validate_max_tokens)] = Field(
        default=1000, description="Maximum tokens to generate"
    )
    prompt: Optional[str] = Field(None, description="User prompt template")
    system_prompt: Optional[str] = Field(None, description="System prompt")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v or not v.strip():
            raise ValueError("model cannot be empty")
        return v.strip()

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: Optional[str]) -> Optional[str]:
        """Validate prompt is not empty if provided."""
        if v is not None and not v.strip():
            raise ValueError("prompt cannot be empty if provided")
        return v.strip() if v else v

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v: Optional[str]) -> Optional[str]:
        """Validate system_prompt is not empty if provided."""
        if v is not None and not v.strip():
            raise ValueError("system_prompt cannot be empty if provided")
        return v.strip() if v else v


class ReactNodeConfig(BaseNodeConfig):
    """Configuration for React nodes."""

    tools: List[str] = Field(default_factory=list, description="List of tool names")
    reasoning_prompt: Optional[str] = Field(
        None, description="Reasoning prompt for the agent"
    )
    system_prompt: Optional[str] = Field(
        None, description="System prompt for the agent"
    )
    max_iterations: Annotated[int, AfterValidator(validate_max_iterations)] = Field(
        default=5, description="Maximum reasoning iterations"
    )

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v: List[str]) -> List[str]:
        """Validate tool names are not empty."""
        for tool in v:
            if not tool or not tool.strip():
                raise ValueError("tool names cannot be empty")
        return [tool.strip() for tool in v]

    @field_validator("reasoning_prompt")
    @classmethod
    def validate_reasoning_prompt(cls, v: Optional[str]) -> Optional[str]:
        """Validate reasoning_prompt is not empty if provided."""
        if v is not None and not v.strip():
            raise ValueError("reasoning_prompt cannot be empty if provided")
        return v.strip() if v else v

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v: Optional[str]) -> Optional[str]:
        """Validate system_prompt is not empty if provided."""
        if v is not None and not v.strip():
            raise ValueError("system_prompt cannot be empty if provided")
        return v.strip() if v else v


# Union type for all node configurations
NodeConfig = LLMNodeConfig | ReactNodeConfig
