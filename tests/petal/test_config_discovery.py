import os
from unittest.mock import patch

import pytest
import yaml
from langchain_core.tools import BaseTool
from petal.core.discovery.config import ConfigDiscovery

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "../fixtures/")


@pytest.mark.asyncio
async def test_config_discovery_finds_tool(tmp_path):
    # Arrange: Write a valid YAML config with a tool definition
    config_path = tmp_path / "tools.yaml"
    tool_yaml = {
        "tools": {
            "my_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "echo_tool",
            }
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(tool_yaml, f)

    # Act
    discovery = ConfigDiscovery(config_locations=[str(config_path)])
    tool = await discovery.discover("my_tool")

    # Assert
    assert tool is not None
    assert isinstance(tool, BaseTool)
    assert tool.name == "my_tool"
    assert callable(tool.run)


@pytest.mark.asyncio
async def test_config_discovery_tool_not_found(tmp_path):
    # Arrange: Write a valid YAML config with a different tool
    config_path = tmp_path / "tools.yaml"
    tool_yaml = {
        "tools": {
            "other_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "echo_tool",
            }
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(tool_yaml, f)

    # Act
    discovery = ConfigDiscovery(config_locations=[str(config_path)])
    tool = await discovery.discover("missing_tool")

    # Assert
    assert tool is None


@pytest.mark.asyncio
async def test_config_discovery_malformed_yaml(tmp_path):
    # Arrange: Write an invalid YAML config
    config_path = tmp_path / "tools.yaml"
    with open(config_path, "w") as f:
        f.write(": this is not valid yaml ::\nfoo: [unclosed\n")

    # Act & Assert
    discovery = ConfigDiscovery(config_locations=[str(config_path)])
    with pytest.raises(ValueError):
        await discovery.discover("any_tool")


@pytest.mark.asyncio
async def test_config_discovery_dynamic_module_loading(tmp_path):
    # Arrange: Write a YAML config referencing a tool in a module not yet imported
    config_path = tmp_path / "tools.yaml"
    tool_yaml = {
        "tools": {
            "dynamic_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "echo_tool",
            }
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(tool_yaml, f)

    # Act
    discovery = ConfigDiscovery(config_locations=[str(config_path)])
    tool = await discovery.discover("dynamic_tool")

    # Assert
    assert tool is not None
    assert isinstance(tool, BaseTool)
    assert tool.name == "dynamic_tool"
    assert callable(tool.run)


@pytest.mark.asyncio
async def test_config_discovery_directory_scanning(tmp_path):
    """Test scanning directories for YAML files."""
    # Arrange: Create a directory with YAML files
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Create a valid YAML file
    config_file = config_dir / "tools.yaml"
    tool_yaml = {
        "tools": {
            "dir_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "echo_tool",
            }
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(tool_yaml, f)

    # Create a non-YAML file (should be ignored)
    non_yaml_file = config_dir / "readme.txt"
    with open(non_yaml_file, "w") as f:
        f.write("This is not a YAML file")

    # Act
    discovery = ConfigDiscovery(config_locations=[str(config_dir)])
    tool = await discovery.discover("dir_tool")

    # Assert
    assert tool is not None
    assert isinstance(tool, BaseTool)
    assert tool.name == "dir_tool"


@pytest.mark.asyncio
async def test_config_discovery_file_not_found(tmp_path):
    """Test handling of non-existent files."""
    # Arrange: Use a non-existent file path
    non_existent_file = tmp_path / "nonexistent.yaml"

    # Act
    discovery = ConfigDiscovery(config_locations=[str(non_existent_file)])
    tool = await discovery.discover("any_tool")

    # Assert: Should handle gracefully and return None
    assert tool is None


@pytest.mark.asyncio
async def test_config_discovery_file_read_error(tmp_path):
    """Test handling of file read errors."""
    # Arrange: Create a file that can't be read
    config_path = tmp_path / "tools.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"tools": {}}, f)

    # Make the file unreadable
    os.chmod(config_path, 0o000)

    try:
        # Act
        discovery = ConfigDiscovery(config_locations=[str(config_path)])
        tool = await discovery.discover("any_tool")

        # Assert: Should handle gracefully and return None
        assert tool is None
    finally:
        # Restore permissions for cleanup
        os.chmod(config_path, 0o644)


@pytest.mark.asyncio
async def test_config_discovery_missing_module_or_function(tmp_path):
    """Test handling of configs with missing module or function."""
    # Arrange: Write a config with missing module
    config_path = tmp_path / "tools.yaml"
    tool_yaml = {
        "tools": {
            "incomplete_tool": {
                "function": "echo_tool"
                # Missing module
            }
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(tool_yaml, f)

    # Act & Assert
    discovery = ConfigDiscovery(config_locations=[str(config_path)])
    with pytest.raises(ValueError, match="missing module or function"):
        await discovery.discover("incomplete_tool")


@pytest.mark.asyncio
async def test_config_discovery_import_error(tmp_path):
    """Test handling of import errors."""
    # Arrange: Write a config with non-existent module
    config_path = tmp_path / "tools.yaml"
    tool_yaml = {
        "tools": {
            "import_error_tool": {
                "module": "nonexistent.module",
                "function": "some_function",
            }
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(tool_yaml, f)

    # Act & Assert
    discovery = ConfigDiscovery(config_locations=[str(config_path)])
    with pytest.raises(ValueError, match="Could not load tool"):
        await discovery.discover("import_error_tool")


@pytest.mark.asyncio
async def test_config_discovery_attribute_error(tmp_path):
    """Test handling of attribute errors (function not found in module)."""
    # Arrange: Write a config with non-existent function
    config_path = tmp_path / "tools.yaml"
    tool_yaml = {
        "tools": {
            "attr_error_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "nonexistent_function",
            }
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(tool_yaml, f)

    # Act & Assert
    discovery = ConfigDiscovery(config_locations=[str(config_path)])
    with pytest.raises(ValueError, match="Could not load tool"):
        await discovery.discover("attr_error_tool")


@pytest.mark.asyncio
async def test_config_discovery_general_exception_handling():
    """Test handling of general exceptions during discovery."""
    # Arrange: Create a discovery instance
    discovery = ConfigDiscovery(config_locations=["/invalid/path"])

    # Mock _load_config_files to raise a general exception
    with patch.object(
        discovery, "_load_config_files", side_effect=Exception("General error")
    ):
        # Act
        tool = await discovery.discover("any_tool")

        # Assert: Should handle gracefully and return None
        assert tool is None


@pytest.mark.asyncio
async def test_config_discovery_value_error_propagation():
    """Test that ValueError exceptions are properly propagated."""
    # Arrange: Create a discovery instance
    discovery = ConfigDiscovery(config_locations=["/invalid/path"])

    # Mock _load_config_files to raise a ValueError
    with (
        patch.object(
            discovery, "_load_config_files", side_effect=ValueError("YAML error")
        ),
        pytest.raises(ValueError, match="YAML error"),
    ):
        # Act & Assert: ValueError should be re-raised
        await discovery.discover("any_tool")


@pytest.mark.asyncio
async def test_config_discovery_load_config_from_file_exception():
    """Test the general exception handling in _load_config_from_file."""
    # Arrange: Create a discovery instance
    discovery = ConfigDiscovery()

    # Mock open to raise an exception that's not yaml.YAMLError
    with patch("builtins.open", side_effect=OSError("File system error")):
        # Act
        result = await discovery._load_config_from_file("dummy.yaml")

        # Assert: Should return None for general exceptions
        assert result is None


@pytest.mark.asyncio
async def test_config_discovery_load_tool_from_config_regular_function(tmp_path):
    """Test loading a regular function (not BaseTool) from config."""
    # Arrange: Write a config for a regular function
    config_path = tmp_path / "tools.yaml"
    tool_yaml = {
        "tools": {
            "regular_function_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "regular_function",
            }
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(tool_yaml, f)

    # Act
    discovery = ConfigDiscovery(config_locations=[str(config_path)])
    tool = await discovery.discover("regular_function_tool")

    # Assert: Should return the function directly (not wrapped in BaseTool)
    assert tool is not None
    assert callable(tool)
    # Test that it's the actual function, not a BaseTool
    result = tool("test")
    assert result == "Regular: test"


@pytest.mark.asyncio
async def test_config_discovery_file_open_exception_via_discover(tmp_path):
    """Test that a general exception in _load_config_from_file via discover returns None."""
    # Arrange: Use a file path that will trigger an OSError
    config_path = tmp_path / "tools.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            {
                "tools": {
                    "foo": {
                        "module": "tests.fixtures.sample_tools",
                        "function": "echo_tool",
                    }
                }
            },
            f,
        )

    # Patch open to raise OSError only when this file is opened
    orig_open = open

    def open_side_effect(path, *args, **kwargs):
        if str(path).endswith("tools.yaml"):
            raise OSError("File system error")
        return orig_open(path, *args, **kwargs)

    with patch("builtins.open", side_effect=open_side_effect):
        discovery = ConfigDiscovery(config_locations=[str(config_path)])
        tool = await discovery.discover("foo")
        assert tool is None


@pytest.mark.asyncio
async def test_config_discovery_broken_symlink(tmp_path):
    """Test that a broken symlink triggers the generic exception handler in _load_config_from_file via discover."""
    # Arrange: Create a broken symlink
    config_path = tmp_path / "broken.yaml"
    config_path.symlink_to(tmp_path / "does_not_exist.yaml")

    # Act
    discovery = ConfigDiscovery(config_locations=[str(config_path)])
    tool = await discovery.discover("foo")
    # Assert: Should return None
    assert tool is None


@pytest.mark.asyncio
async def test_config_discovery_directory_with_broken_and_valid_yaml(tmp_path):
    """Test that a directory with a broken YAML file and a valid YAML file skips the broken one and loads the valid tool."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Create a broken symlink (will cause OSError)
    broken_file = config_dir / "broken.yaml"
    broken_file.symlink_to(config_dir / "does_not_exist.yaml")

    # Create a valid YAML file
    valid_file = config_dir / "tools.yaml"
    tool_yaml = {
        "tools": {
            "valid_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "echo_tool",
            }
        }
    }
    with open(valid_file, "w") as f:
        yaml.dump(tool_yaml, f)

    # Act
    discovery = ConfigDiscovery(config_locations=[str(config_dir)])
    tool = await discovery.discover("valid_tool")

    # Assert: Should skip the broken file and still find the valid tool
    assert tool is not None
    assert isinstance(tool, BaseTool)
    assert tool.name == "valid_tool"


@pytest.mark.asyncio
async def test_config_discovery_directory_permission_error(tmp_path):
    """Test that PermissionError during directory scanning triggers the except Exception block."""
    # Arrange: Create a directory
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Create a valid YAML file in the directory
    valid_file = config_dir / "tools.yaml"
    tool_yaml = {
        "tools": {
            "permission_test_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "echo_tool",
            }
        }
    }
    with open(valid_file, "w") as f:
        yaml.dump(tool_yaml, f)

    # Mock os.listdir to raise PermissionError for this specific directory
    original_listdir = os.listdir

    def mock_listdir(path):
        if str(path) == str(config_dir):
            raise PermissionError("Access denied")
        return original_listdir(path)

    with patch("os.listdir", side_effect=mock_listdir):
        # Act
        discovery = ConfigDiscovery(config_locations=[str(config_dir)])
        tool = await discovery.discover("permission_test_tool")

        # Assert: Should handle the exception gracefully and return None
        # since the directory scan failed
        assert tool is None
