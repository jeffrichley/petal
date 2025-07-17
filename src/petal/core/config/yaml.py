"""YAML configuration models for node loading."""

import re
from typing import Annotated, Any, Dict, List, Optional, Type

from pydantic import AfterValidator, BaseModel, Field, field_validator


class TypeResolver:
    """Resolves string type names to Python types."""

    _type_mapping = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": List,
        "dict": Dict,
        "tuple": tuple,
        "set": set,
        "bytes": bytes,
        "None": type(None),
    }

    @classmethod
    def resolve_type(cls, type_name: str) -> Type:
        """Resolve string type name to Python type.

        Args:
            type_name: The string representation of the type

        Returns:
            The Python type

        Raises:
            ValueError: If the type name is not supported
        """
        if type_name not in cls._type_mapping:
            raise ValueError(f"Unsupported type: {type_name}")
        return cls._type_mapping[type_name]

    @classmethod
    def is_valid_type(cls, type_name: str) -> bool:
        """Check if a type name is valid.

        Args:
            type_name: The string representation of the type

        Returns:
            True if the type is supported, False otherwise
        """
        return type_name in cls._type_mapping


def validate_field_name(v: str) -> str:
    """Validate field name is a valid Python identifier."""
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
        raise ValueError(f"Invalid field name: {v}")
    return v


def validate_node_type(v: str) -> str:
    """Validate node type is supported."""
    valid_types = ["llm", "react", "custom"]
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

    fields: Dict[str, str] = Field(
        default_factory=dict, description="Field name to type string mapping"
    )
    required_fields: List[str] = Field(
        default_factory=list, description="List of required field names"
    )
    optional_fields: Dict[str, Any] = Field(
        default_factory=dict, description="Optional fields with default values"
    )
    nested_schemas: Dict[str, "StateSchemaConfig"] = Field(
        default_factory=dict, description="Nested schema definitions"
    )

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate field names and types."""
        for field_name, type_name in v.items():
            # Validate field name
            validate_field_name(field_name)
            # Validate type name
            if not TypeResolver.is_valid_type(type_name):
                raise ValueError(f"Unsupported type: {type_name}")
        return v

    @field_validator("required_fields")
    @classmethod
    def validate_required_fields(cls, v: List[str], info) -> List[str]:
        """Validate required fields exist in fields."""
        fields = info.data.get("fields", {})
        for field_name in v:
            if field_name not in fields:
                raise ValueError(f"Required field not found in fields: {field_name}")
        return v

    @field_validator("optional_fields")
    @classmethod
    def validate_optional_fields(cls, v: Dict[str, Any], info) -> Dict[str, Any]:
        """Validate optional fields exist in fields."""
        fields = info.data.get("fields", {})
        for field_name in v:
            if field_name not in fields:
                raise ValueError(f"Optional field not found in fields: {field_name}")
        return v

    def get_resolved_fields(self) -> Dict[str, Type]:
        """Get fields with resolved Python types.

        Returns:
            Dictionary mapping field names to Python types
        """
        return {
            name: TypeResolver.resolve_type(type_name)
            for name, type_name in self.fields.items()
        }


class ValidationConfig(BaseModel):
    """Configuration for input/output validation."""

    input_schema: Optional[StateSchemaConfig] = Field(
        None, description="Input validation schema"
    )
    output_schema: Optional[StateSchemaConfig] = Field(
        None, description="Output validation schema"
    )


class ToolDiscoveryConfig(BaseModel):
    """Configuration for YAML nodes."""

    enabled: bool = Field(default=True, description="Whether tool discovery is enabled")
    folders: Optional[List[str]] = Field(
        default=None, description="Custom folders to scan for tools"
    )
    config_locations: Optional[List[str]] = Field(
        default=None, description="Custom config file locations to scan"
    )
    exclude_patterns: Optional[List[str]] = Field(
        default=None, description="Custom exclusion patterns"
    )


class BaseNodeConfig(BaseModel):
    """Base configuration for all node types."""

    type: Annotated[str, AfterValidator(validate_node_type)] = Field(
        ..., description="Type of node (llm, react)"
    )
    name: str = Field(..., description="Name of the node")
    description: Optional[str] = Field(
        default=None, description="Description of the node"
    )
    enabled: bool = Field(default=True, description="Whether the node is enabled")
    state_schema: Optional[StateSchemaConfig] = Field(
        default=None, description="State schema definition for this node"
    )
    input_schema: Optional[StateSchemaConfig] = Field(
        default=None, description="Input validation schema"
    )
    output_schema: Optional[StateSchemaConfig] = Field(
        default=None, description="Output validation schema"
    )
    tool_discovery: Optional[ToolDiscoveryConfig] = Field(
        default=None, description="Tool discovery configuration"
    )

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
    prompt: Optional[str] = Field(default=None, description="User prompt template")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")

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
        default=None, description="Reasoning prompt for the agent"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for the agent"
    )
    max_iterations: Annotated[int, AfterValidator(validate_max_iterations)] = Field(
        default=5, description="Maximum reasoning iterations"
    )
    mcp_servers: Optional[Dict[str, Any]] = Field(
        default=None,
        description="MCP server configurations for tool registry integration",
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


class CustomNodeConfig(BaseNodeConfig):
    """Configuration for Custom nodes."""

    function_path: str = Field(..., description="Python import path to function")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Function parameters"
    )
    validation: Optional[ValidationConfig] = Field(
        default=None, description="Input/output validation configuration"
    )

    @field_validator("function_path")
    @classmethod
    def validate_function_path(cls, v: str) -> str:
        """Validate function path is not empty."""
        if not v or not v.strip():
            raise ValueError("function_path cannot be empty")
        return v.strip()


# Union type for all node configurations
NodeConfig = LLMNodeConfig | ReactNodeConfig | CustomNodeConfig
