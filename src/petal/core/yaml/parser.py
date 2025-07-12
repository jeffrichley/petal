import os
from typing import Any, Dict

import yaml

from petal.core.config.yaml import LLMNodeConfig


class YAMLFileNotFoundError(Exception):
    pass


class YAMLParseError(Exception):
    pass


class YAMLNodeParser:
    def __init__(self):
        pass

    def load_yaml_file(self, path: str) -> Dict[str, Any]:
        raise NotImplementedError

    def detect_node_type(self, yaml_data: Dict[str, Any]) -> str:
        raise NotImplementedError

    def validate_yaml_schema(self, yaml_data: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def parse_node_config(self, path: str):
        if not os.path.exists(path):
            raise YAMLFileNotFoundError(f"YAML file not found: {path}")
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise YAMLParseError(f"Invalid YAML syntax in {path}: {e}") from e
        if not isinstance(data, dict):
            raise YAMLParseError(f"YAML root must be a mapping/object in {path}")
        node_type = data.get("type")
        if node_type == "llm":
            return LLMNodeConfig(**data)
        raise ValueError(f"Unsupported node type: {node_type}")
