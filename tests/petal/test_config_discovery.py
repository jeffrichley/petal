import os

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
async def test_config_discovery_general_exception_handling(tmp_path):
    """Test handling of general exceptions during discovery."""
    # Arrange: Create a directory with a file that will cause a real exception
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Create a file that will cause an OSError when trying to read it
    config_file = config_dir / "tools.yaml"
    with open(config_file, "w") as f:
        f.write(
            "tools:\n  test_tool:\n    module: tests.fixtures.sample_tools\n    function: echo_tool\n"
        )

    # Make the file unreadable to trigger a real exception
    os.chmod(config_file, 0o000)

    try:
        # Act
        discovery = ConfigDiscovery(config_locations=[str(config_dir)])
        tool = await discovery.discover("test_tool")

        # Assert: Should handle gracefully and return None
        assert tool is None
    finally:
        # Restore permissions for cleanup
        os.chmod(config_file, 0o644)


@pytest.mark.asyncio
async def test_config_discovery_value_error_propagation(tmp_path):
    """Test that ValueError exceptions are properly propagated."""
    # Arrange: Create a file with malformed YAML that will cause a real ValueError
    config_path = tmp_path / "tools.yaml"
    with open(config_path, "w") as f:
        f.write(
            "tools:\n  test_tool:\n    module: tests.fixtures.sample_tools\n    function: echo_tool\n    unclosed: [list\n"
        )

    # Act & Assert: ValueError should be re-raised from real YAML parsing
    discovery = ConfigDiscovery(config_locations=[str(config_path)])
    with pytest.raises(ValueError, match="Invalid YAML"):
        await discovery.discover("test_tool")


@pytest.mark.asyncio
async def test_config_discovery_load_config_from_file_exception(tmp_path):
    """Test the general exception handling in _load_config_from_file."""
    # Arrange: Create a file that will cause a real OSError
    config_path = tmp_path / "tools.yaml"
    with open(config_path, "w") as f:
        f.write(
            "tools:\n  test_tool:\n    module: tests.fixtures.sample_tools\n    function: echo_tool\n"
        )

    # Make the file unreadable to trigger a real OSError
    os.chmod(config_path, 0o000)

    try:
        # Act
        discovery = ConfigDiscovery()
        result = await discovery._load_config_from_file(str(config_path))

        # Assert: Should return None for general exceptions
        assert result is None
    finally:
        # Restore permissions for cleanup
        os.chmod(config_path, 0o644)


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
    # Arrange: Create a file that will cause a real OSError
    config_path = tmp_path / "tools.yaml"
    with open(config_path, "w") as f:
        f.write(
            "tools:\n  test_tool:\n    module: tests.fixtures.sample_tools\n    function: echo_tool\n"
        )

    # Make the file unreadable to trigger a real OSError
    os.chmod(config_path, 0o000)

    try:
        # Act
        discovery = ConfigDiscovery(config_locations=[str(config_path)])
        tool = await discovery.discover("test_tool")

        # Assert: Should return None due to real file system error
        assert tool is None
    finally:
        # Restore permissions for cleanup
        os.chmod(config_path, 0o644)


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

    # Make the directory unreadable to trigger a real PermissionError
    os.chmod(config_dir, 0o000)

    try:
        # Act
        discovery = ConfigDiscovery(config_locations=[str(config_dir)])
        tool = await discovery.discover("permission_test_tool")

        # Assert: Should handle the exception gracefully and return None
        # since the directory scan failed due to real permission error
        assert tool is None
    finally:
        # Restore permissions for cleanup
        os.chmod(config_dir, 0o755)


@pytest.mark.asyncio
async def test_config_discovery_integration_comprehensive(tmp_path):
    """Comprehensive integration test for ConfigDiscovery with real file system operations."""
    # Arrange: Create a complex directory structure with multiple config files
    config_root = tmp_path / "configs"
    config_root.mkdir()

    # Create subdirectories
    tools_dir = config_root / "tools"
    tools_dir.mkdir()

    agents_dir = config_root / "agents"
    agents_dir.mkdir()

    # Create multiple YAML files with different tools
    tools_file = tools_dir / "tools.yaml"
    tools_yaml = {
        "tools": {
            "echo_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "echo_tool",
            },
            "regular_function_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "regular_function",
            },
        }
    }
    with open(tools_file, "w") as f:
        yaml.dump(tools_yaml, f)

    # Create another tools file in a different location
    alt_tools_file = config_root / "alt_tools.yml"
    alt_tools_yaml = {
        "tools": {
            "alt_echo_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "echo_tool",
            }
        }
    }
    with open(alt_tools_file, "w") as f:
        yaml.dump(alt_tools_yaml, f)

    # Create a non-YAML file (should be ignored)
    readme_file = config_root / "README.txt"
    with open(readme_file, "w") as f:
        f.write("This is not a YAML file")

    # Act: Test discovery with multiple config locations
    discovery = ConfigDiscovery(
        config_locations=[
            str(tools_dir),  # Directory scanning
            str(alt_tools_file),  # Direct file
        ]
    )

    # Test finding tools from different sources
    echo_tool = await discovery.discover("echo_tool")
    assert echo_tool is not None
    assert isinstance(echo_tool, BaseTool)
    assert echo_tool.name == "echo_tool"

    regular_function_tool = await discovery.discover("regular_function_tool")
    assert regular_function_tool is not None
    assert callable(regular_function_tool)
    assert regular_function_tool("test") == "Regular: test"

    alt_echo_tool = await discovery.discover("alt_echo_tool")
    assert alt_echo_tool is not None
    assert isinstance(alt_echo_tool, BaseTool)
    assert alt_echo_tool.name == "alt_echo_tool"

    # Test that non-existent tools return None
    missing_tool = await discovery.discover("missing_tool")
    assert missing_tool is None


@pytest.mark.asyncio
async def test_config_discovery_malformed_yaml_raises_value_error(tmp_path):
    """Test that malformed YAML files raise ValueError and prevent discovery."""
    # Arrange: Create a directory with a malformed YAML file
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Create a file with malformed YAML
    bad_file = config_dir / "bad.yaml"
    with open(bad_file, "w") as f:
        f.write(
            "tools:\n  bad_tool:\n    module: tests.fixtures.sample_tools\n    function: echo_tool\n    unclosed: [list\n"
        )

    # Act & Assert: Discovery should raise ValueError when encountering malformed YAML
    discovery = ConfigDiscovery(config_locations=[str(config_dir)])
    with pytest.raises(ValueError, match="Invalid YAML"):
        await discovery.discover("any_tool")


@pytest.mark.asyncio
async def test_config_discovery_cache_behavior(tmp_path):
    """Test that ConfigDiscovery properly caches config files and doesn't reload them."""
    # Arrange: Create a config file
    config_path = tmp_path / "tools.yaml"
    tool_yaml = {
        "tools": {
            "cached_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "echo_tool",
            }
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(tool_yaml, f)

    # Act: Create discovery instance and discover a tool
    discovery = ConfigDiscovery(config_locations=[str(config_path)])

    # First discovery should load the config
    tool1 = await discovery.discover("cached_tool")
    assert tool1 is not None

    # Modify the file after first discovery
    modified_yaml = {
        "tools": {
            "cached_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "regular_function",  # Changed function
            }
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(modified_yaml, f)

    # Second discovery should use cached config, not the modified file
    tool2 = await discovery.discover("cached_tool")
    assert tool2 is not None
    assert tool2.name == "cached_tool"

    # The tool should still be the original echo_tool, not the modified regular_function
    # This tests that the cache is working and preventing reloading
    assert isinstance(tool2, BaseTool)  # Should still be BaseTool, not regular function


@pytest.mark.asyncio
async def test_config_discovery_error_recovery(tmp_path):
    """Test that ConfigDiscovery can recover from some errors but not malformed YAML or import errors."""
    # Arrange: Create a directory with multiple files, some with errors
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Valid file
    valid_file = config_dir / "valid.yaml"
    valid_yaml = {
        "tools": {
            "valid_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "echo_tool",
            }
        }
    }
    with open(valid_file, "w") as f:
        yaml.dump(valid_yaml, f)

    # File with attribute error (function doesn't exist in module)
    attr_error_file = config_dir / "attr_error.yaml"
    attr_error_yaml = {
        "tools": {
            "attr_error_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "nonexistent_function",
            }
        }
    }
    with open(attr_error_file, "w") as f:
        yaml.dump(attr_error_yaml, f)

    # Act: Test discovery
    discovery = ConfigDiscovery(config_locations=[str(config_dir)])

    # Should find the valid tool
    valid_tool = await discovery.discover("valid_tool")
    assert valid_tool is not None
    assert isinstance(valid_tool, BaseTool)
    assert valid_tool.name == "valid_tool"

    # Should raise ValueError for import/attribute errors
    with pytest.raises(ValueError, match="Could not load tool"):
        await discovery.discover("attr_error_tool")


@pytest.mark.asyncio
async def test_config_discovery_import_error_raises_value_error(tmp_path):
    """Test that import errors raise ValueError and prevent discovery."""
    # Arrange: Create a directory with a file that has import errors
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # File with import error
    import_error_file = config_dir / "import_error.yaml"
    import_error_yaml = {
        "tools": {
            "import_error_tool": {
                "module": "nonexistent.module",
                "function": "some_function",
            }
        }
    }
    with open(import_error_file, "w") as f:
        yaml.dump(import_error_yaml, f)

    # Act & Assert: Discovery should raise ValueError when encountering import errors
    discovery = ConfigDiscovery(config_locations=[str(config_dir)])
    with pytest.raises(ValueError, match="Could not load tool"):
        await discovery.discover("import_error_tool")


@pytest.mark.asyncio
async def test_config_discovery_malformed_yaml_prevents_processing(tmp_path):
    """Test that malformed YAML prevents processing of other files in the same directory."""
    # Arrange: Create a directory with both valid and malformed files
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Valid file
    valid_file = config_dir / "valid.yaml"
    valid_yaml = {
        "tools": {
            "valid_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "echo_tool",
            }
        }
    }
    with open(valid_file, "w") as f:
        yaml.dump(valid_yaml, f)

    # File with malformed YAML (this should prevent processing of the entire directory)
    malformed_file = config_dir / "malformed.yaml"
    with open(malformed_file, "w") as f:
        f.write(
            "tools:\n  malformed_tool:\n    module: tests.fixtures.sample_tools\n    function: echo_tool\n    unclosed: [list\n"
        )

    # Act & Assert: Discovery should raise ValueError when encountering malformed YAML
    discovery = ConfigDiscovery(config_locations=[str(config_dir)])
    with pytest.raises(ValueError, match="Invalid YAML"):
        await discovery.discover("valid_tool")


@pytest.mark.asyncio
async def test_config_discovery_default_locations(tmp_path):
    """Test that ConfigDiscovery works with default config locations."""
    # Arrange: Create a file in one of the default locations
    tools_file = tmp_path / "tools.yaml"
    tool_yaml = {
        "tools": {
            "default_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "echo_tool",
            }
        }
    }
    with open(tools_file, "w") as f:
        yaml.dump(tool_yaml, f)

    # Change to the temp directory to test default locations
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Act: Create discovery with default locations
        discovery = ConfigDiscovery()  # Uses DEFAULT_CONFIG_LOCATIONS

        # Should find the tool in the default location
        tool = await discovery.discover("default_tool")
        assert tool is not None
        assert isinstance(tool, BaseTool)
        assert tool.name == "default_tool"

    finally:
        os.chdir(original_cwd)


@pytest.mark.asyncio
async def test_config_discovery_tool_name_override(tmp_path):
    """Test that tools are properly named according to the config, not the original function name."""
    # Arrange: Create a config file with a tool that has a different name than the function
    config_path = tmp_path / "tools.yaml"
    tool_yaml = {
        "tools": {
            "custom_named_tool": {  # Custom name in config
                "module": "tests.fixtures.sample_tools",
                "function": "echo_tool",  # Original function name
            }
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(tool_yaml, f)

    # Act
    discovery = ConfigDiscovery(config_locations=[str(config_path)])
    tool = await discovery.discover("custom_named_tool")

    # Assert: The tool should have the name from the config, not the original function
    assert tool is not None
    assert isinstance(tool, BaseTool)
    assert (
        tool.name == "custom_named_tool"
    )  # Should be the config name, not "echo_tool"

    # The tool should still work correctly
    result = await tool.ainvoke({"text": "test message"})
    assert result == "test message"


@pytest.mark.asyncio
async def test_config_discovery_real_world_usage(tmp_path):
    """Test ConfigDiscovery in a realistic real-world scenario."""
    # Arrange: Create a realistic project structure
    project_root = tmp_path / "my_project"
    project_root.mkdir()

    # Create config directory
    config_dir = project_root / "configs"
    config_dir.mkdir()

    # Create tools directory
    tools_dir = project_root / "tools"
    tools_dir.mkdir()

    # Create a tools.py file with actual tools
    tools_file = tools_dir / "tools.py"
    with open(tools_file, "w") as f:
        f.write(
            '''from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the database for information."""
    return f"Searching database for: {query}"

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to the specified recipient."""
    return f"Email sent to {recipient}: {subject}"

@tool
def calculate_tax(amount: float, rate: float) -> float:
    """Calculate tax for a given amount and rate."""
    return amount * rate / 100
'''
        )

    # Create a YAML config file
    config_file = config_dir / "tools.yaml"
    config_yaml = {
        "tools": {
            "db_search": {
                "module": "tools.tools",
                "function": "search_database",
            },
            "email_sender": {
                "module": "tools.tools",
                "function": "send_email",
            },
            "tax_calculator": {
                "module": "tools.tools",
                "function": "calculate_tax",
            },
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(config_yaml, f)

    # Add the tools directory to Python path for import
    import sys

    sys.path.insert(0, str(project_root))

    try:
        # Act: Test discovery
        discovery = ConfigDiscovery(config_locations=[str(config_file)])

        # Test discovering tools
        db_search = await discovery.discover("db_search")
        assert db_search is not None
        assert isinstance(db_search, BaseTool)
        assert db_search.name == "db_search"

        email_sender = await discovery.discover("email_sender")
        assert email_sender is not None
        assert isinstance(email_sender, BaseTool)
        assert email_sender.name == "email_sender"

        tax_calculator = await discovery.discover("tax_calculator")
        assert tax_calculator is not None
        assert isinstance(tax_calculator, BaseTool)
        assert tax_calculator.name == "tax_calculator"
        tax_result = float(await tax_calculator.ainvoke({"amount": 100, "rate": 10}))
        assert tax_result == 10.0

        # Test that tools work correctly
        db_result = await db_search.ainvoke({"query": "user data"})
        assert db_result == "Searching database for: user data"

        email_result = await email_sender.ainvoke(
            {"recipient": "user@example.com", "subject": "Test", "body": "Hello"}
        )
        assert email_result == "Email sent to user@example.com: Test"

        # Test that non-existent tools return None
        missing_tool = await discovery.discover("missing_tool")
        assert missing_tool is None

    finally:
        # Clean up Python path
        sys.path.remove(str(project_root))


@pytest.mark.asyncio
async def test_config_discovery_specific_exception_handler_line_49(tmp_path):
    """Test specifically targets the 'return None' line on line 49 of config.py.

    This test ensures that when a general exception (not ValueError) occurs during
    the discover method, the code gracefully returns None instead of crashing.
    """
    # Arrange: Create a scenario that will trigger the general exception handler
    # We'll create a file that exists but will cause an exception when processed
    config_path = tmp_path / "tools.yaml"

    # Create a file with valid YAML structure but invalid content that will cause
    # an exception during processing (not a ValueError, but a general exception)
    with open(config_path, "w") as f:
        f.write(
            "tools:\n  test_tool:\n    module: tests.fixtures.sample_tools\n    function: echo_tool\n"
        )

    # Make the file unreadable to trigger a real OSError during processing
    os.chmod(config_path, 0o000)

    try:
        # Act: This should trigger the general exception handler and return None
        discovery = ConfigDiscovery(config_locations=[str(config_path)])
        result = await discovery.discover("test_tool")

        # Assert: The specific line "return None" on line 49 should be executed
        assert (
            result is None
        ), "Expected None to be returned from general exception handler"

    finally:
        # Restore permissions for cleanup
        os.chmod(config_path, 0o644)


@pytest.mark.asyncio
async def test_config_discovery_exception_handler_coverage(tmp_path):
    """Test to ensure the exception handler on line 49 gets proper coverage.

    This test creates multiple scenarios that should trigger the general exception
    handler to ensure comprehensive coverage of the 'return None' line.
    """
    # Test scenario 1: Broken symlink
    broken_symlink = tmp_path / "broken.yaml"
    broken_symlink.symlink_to(tmp_path / "does_not_exist.yaml")

    discovery = ConfigDiscovery(config_locations=[str(broken_symlink)])
    result1 = await discovery.discover("any_tool")
    assert result1 is None, "Broken symlink should return None"

    # Test scenario 2: Directory permission error
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Create a valid file in the directory
    valid_file = config_dir / "tools.yaml"
    with open(valid_file, "w") as f:
        f.write(
            "tools:\n  test_tool:\n    module: tests.fixtures.sample_tools\n    function: echo_tool\n"
        )

    # Make directory unreadable
    os.chmod(config_dir, 0o000)

    try:
        discovery2 = ConfigDiscovery(config_locations=[str(config_dir)])
        result2 = await discovery2.discover("test_tool")
        assert result2 is None, "Directory permission error should return None"
    finally:
        os.chmod(config_dir, 0o755)

    # Test scenario 3: File permission error
    config_file = tmp_path / "permission_error.yaml"
    with open(config_file, "w") as f:
        f.write(
            "tools:\n  test_tool:\n    module: tests.fixtures.sample_tools\n    function: echo_tool\n"
        )

    os.chmod(config_file, 0o000)

    try:
        discovery3 = ConfigDiscovery(config_locations=[str(config_file)])
        result3 = await discovery3.discover("test_tool")
        assert result3 is None, "File permission error should return None"
    finally:
        os.chmod(config_file, 0o644)


@pytest.mark.asyncio
async def test_config_discovery_exception_handler_line_49_with_mock(
    tmp_path, monkeypatch
):
    """Test that specifically targets line 49 using mocking to force the exception handler."""
    # Arrange: Create a valid config file
    config_path = tmp_path / "tools.yaml"
    tool_yaml = {
        "tools": {
            "test_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "echo_tool",
            }
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(tool_yaml, f)

    # Mock _load_config_files to raise a general exception (not ValueError)
    async def mock_load_config_files():
        raise RuntimeError("Simulated general exception")

    discovery = ConfigDiscovery(config_locations=[str(config_path)])
    monkeypatch.setattr(discovery, "_load_config_files", mock_load_config_files)

    # Act: This should trigger the general exception handler and return None
    result = await discovery.discover("test_tool")

    # Assert: The specific line "return None" on line 49 should be executed
    assert result is None, "Expected None to be returned from general exception handler"


@pytest.mark.asyncio
async def test_config_discovery_exception_handler_line_49_with_os_error(tmp_path):
    """Test that specifically targets line 49 by causing an OSError during file operations."""
    # Arrange: Create a scenario that will cause an OSError during _load_config_files
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Create a file that will cause an OSError when trying to list directory contents
    # We'll make the directory unreadable after creating the file
    config_file = config_dir / "tools.yaml"
    with open(config_file, "w") as f:
        f.write(
            "tools:\n  test_tool:\n    module: tests.fixtures.sample_tools\n    function: echo_tool\n"
        )

    # Make the directory unreadable to trigger OSError during os.listdir
    os.chmod(config_dir, 0o000)

    try:
        # Act: This should trigger the general exception handler in _load_config_files
        discovery = ConfigDiscovery(config_locations=[str(config_dir)])
        result = await discovery.discover("test_tool")

        # Assert: The specific line "return None" on line 49 should be executed
        assert (
            result is None
        ), "Expected None to be returned from general exception handler"

    finally:
        # Restore permissions for cleanup
        os.chmod(config_dir, 0o755)


@pytest.mark.asyncio
async def test_config_discovery_exception_handler_complete_coverage(
    tmp_path, monkeypatch
):
    """Test that covers both exception handlers: ValueError (line 46) and general Exception (line 49)."""

    # Test 1: ValueError handling (line 46)
    # Arrange: Create a file with malformed YAML that will cause ValueError
    config_path = tmp_path / "tools.yaml"
    with open(config_path, "w") as f:
        f.write(
            "tools:\n  test_tool:\n    module: tests.fixtures.sample_tools\n    function: echo_tool\n    unclosed: [list\n"
        )

    # Act & Assert: ValueError should be re-raised
    discovery1 = ConfigDiscovery(config_locations=[str(config_path)])
    with pytest.raises(ValueError, match="Invalid YAML"):
        await discovery1.discover("test_tool")

    # Test 2: General Exception handling (line 49)
    # Arrange: Create a valid config file
    config_path2 = tmp_path / "tools2.yaml"
    tool_yaml = {
        "tools": {
            "test_tool": {
                "module": "tests.fixtures.sample_tools",
                "function": "echo_tool",
            }
        }
    }
    with open(config_path2, "w") as f:
        yaml.dump(tool_yaml, f)

    # Mock _load_config_files to raise a general exception (not ValueError)
    async def mock_load_config_files():
        raise RuntimeError("Simulated general exception")

    discovery2 = ConfigDiscovery(config_locations=[str(config_path2)])
    monkeypatch.setattr(discovery2, "_load_config_files", mock_load_config_files)

    # Act: This should trigger the general exception handler and return None
    result = await discovery2.discover("test_tool")

    # Assert: The specific line "return None" on line 49 should be executed
    assert result is None, "Expected None to be returned from general exception handler"
