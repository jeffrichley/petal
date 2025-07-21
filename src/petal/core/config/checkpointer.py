"""Configuration for checkpointer settings."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class CheckpointerConfig(BaseModel):
    """Simple checkpointer configuration."""

    type: Literal["memory", "postgres", "sqlite"] = Field(
        default="memory", description="Type of checkpointer to use"
    )
    """Type of checkpointer to use."""

    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Backend-specific configuration (e.g., database URL for postgres)",
    )
    """Backend-specific configuration (e.g., database URL for postgres)."""

    enabled: bool = Field(default=True, description="Whether checkpointing is enabled")
    """Whether checkpointing is enabled."""
