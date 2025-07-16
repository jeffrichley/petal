"""Tests for folder-based tool discovery."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool
from petal.core.discovery.folder import FolderDiscovery


class TestFolderDiscovery:
    """Test the FolderDiscovery strategy."""

    @pytest.mark.asyncio
    async def test_folder_discovery_initialization(self):
        """Test that FolderDiscovery initializes correctly with defaults."""
        discovery = FolderDiscovery()
        assert discovery.auto_discover is True
        assert discovery.scan_python_path is True
        assert discovery.recursive is True
        assert discovery.folders == FolderDiscovery.DEFAULT_TOOL_FOLDERS
        assert discovery.exclude_patterns == FolderDiscovery.DEFAULT_EXCLUDE_PATTERNS

    @pytest.mark.asyncio
    async def test_folder_discovery_custom_initialization(self):
        """Test that FolderDiscovery can be initialized with custom settings."""
        custom_folders = ["custom_tools/", "my_tools/"]
        custom_excludes = ["*_dev.py", "*_temp.py"]

        discovery = FolderDiscovery(
            auto_discover=False,
            folders=custom_folders,
            exclude_patterns=custom_excludes,
            recursive=False,
            scan_python_path=False,
        )

        assert discovery.auto_discover is False
        assert discovery.scan_python_path is False
        assert discovery.recursive is False
        assert discovery.folders == custom_folders
        assert discovery.exclude_patterns == custom_excludes

    @pytest.mark.asyncio
    async def test_folder_discovery_zero_config_works_out_of_box(self):
        """Test that FolderDiscovery works with zero configuration."""
        discovery = FolderDiscovery()

        # Should not raise any exceptions during initialization
        assert discovery is not None

        # Should be able to call discover without errors
        tool = await discovery.discover("some_tool")
        # Tool may be None, but discovery should work without errors
        assert tool is None or isinstance(tool, BaseTool)

    @pytest.mark.asyncio
    async def test_folder_discovery_handles_invalid_folders(self):
        """Test that FolderDiscovery handles invalid folders gracefully."""
        discovery = FolderDiscovery(
            folders=["/nonexistent/path/", "/another/bad/path/"]
        )

        # Should not raise exceptions for invalid folders
        tool = await discovery.discover("test_tool")
        assert tool is None

    @pytest.mark.asyncio
    async def test_folder_discovery_respects_exclude_patterns(self):
        """Test that FolderDiscovery respects exclude patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file = Path(temp_dir) / "test_tool.py"
            excluded_file = Path(temp_dir) / "test_temp.py"

            test_file.write_text(
                '''
from petal.core.decorators import petaltool

@petaltool("test_tool")
def test_tool():
    """A test tool for discovery."""
    pass
'''
            )

            excluded_file.write_text(
                '''
from petal.core.decorators import petaltool

@petaltool("excluded_tool")
def excluded_tool():
    """An excluded tool for discovery."""
    pass
'''
            )

            discovery = FolderDiscovery(
                folders=[temp_dir], exclude_patterns=["*_temp.py"], auto_discover=False
            )

            # Should find the regular tool
            tool = await discovery.discover("test_tool")
            assert tool is not None

            # Should not find the excluded tool
            excluded_tool = await discovery.discover("excluded_tool")
            assert excluded_tool is None

    @pytest.mark.asyncio
    async def test_folder_discovery_caches_results_for_performance(self):
        """Test that FolderDiscovery caches results for performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a real tool file
            tool_file = Path(temp_dir) / "cached_tool.py"
            tool_file.write_text(
                '''
from petal.core.decorators import petaltool

@petaltool("cached_tool")
def cached_tool():
    """A tool for testing caching."""
    return "cached result"
'''
            )

            discovery = FolderDiscovery(folders=[temp_dir], auto_discover=False)

            # First call should trigger scan and find the tool
            tool1 = await discovery.discover("cached_tool")
            assert tool1 is not None
            assert tool1.name == "cached_tool"

            # Second call should use cache and find the same tool
            tool2 = await discovery.discover("cached_tool")
            assert tool2 is not None
            assert tool2.name == "cached_tool"

            # Should be the same tool instance (from cache)
            assert tool1 is tool2

    # Removed test_folder_discovery_integration_with_registry as it is no longer relevant with real discovery.

    @pytest.mark.asyncio
    async def test_folder_discovery_add_folder_method(self):
        """Test that add_folder method works correctly."""
        discovery = FolderDiscovery(folders=[])
        original_folders = discovery.folders.copy()

        discovery.add_folder("new_tools/")
        assert "new_tools/" in discovery.folders
        assert len(discovery.folders) == len(original_folders) + 1

    @pytest.mark.asyncio
    async def test_folder_discovery_add_exclude_pattern_method(self):
        """Test that add_exclude_pattern method works correctly."""
        discovery = FolderDiscovery(exclude_patterns=[])
        original_patterns = discovery.exclude_patterns.copy()

        discovery.add_exclude_pattern("*_dev.py")
        assert "*_dev.py" in discovery.exclude_patterns
        assert len(discovery.exclude_patterns) == len(original_patterns) + 1

    @pytest.mark.asyncio
    async def test_folder_discovery_enable_disable_auto_discovery(self):
        """Test that enable/disable auto discovery methods work."""
        discovery = FolderDiscovery(auto_discover=False)
        assert discovery.is_auto_discover_enabled() is False

        discovery.enable_auto_discovery()
        assert discovery.is_auto_discover_enabled() is True

        discovery.disable_auto_discovery()
        assert discovery.is_auto_discover_enabled() is False

    @pytest.mark.asyncio
    async def test_folder_discovery_discover_all_method(self):
        """Test that discover_all method returns all tools."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple real tool files
            tool1_file = Path(temp_dir) / "tool1.py"
            tool2_file = Path(temp_dir) / "tool2.py"

            tool1_file.write_text(
                '''
from petal.core.decorators import petaltool

@petaltool("tool1")
def tool1():
    """First test tool."""
    return "result1"
'''
            )

            tool2_file.write_text(
                '''
from petal.core.decorators import petaltool

@petaltool("tool2")
def tool2():
    """Second test tool."""
    return "result2"
'''
            )

            discovery = FolderDiscovery(folders=[temp_dir], auto_discover=False)

            # Should discover all tools
            all_tools = await discovery.discover_all()
            assert len(all_tools) == 2
            assert "tool1" in all_tools
            assert "tool2" in all_tools
            assert all_tools["tool1"].name == "tool1"
            assert all_tools["tool2"].name == "tool2"

            # Verify they are real BaseTool instances
            assert isinstance(all_tools["tool1"], BaseTool)
            assert isinstance(all_tools["tool2"], BaseTool)

    @pytest.mark.asyncio
    async def test_folder_discovery_handles_folder_scanning_exceptions(self):
        """Test that FolderDiscovery gracefully handles folder scanning exceptions."""
        discovery = FolderDiscovery(folders=["/nonexistent/path/"])

        # Mock Path.exists to raise an exception
        with patch("pathlib.Path.exists", side_effect=OSError("Permission denied")):
            # Should not raise exceptions, should handle gracefully
            tools = await discovery._scan_folders()
            assert tools == {}

    @pytest.mark.asyncio
    async def test_folder_discovery_python_path_scanning(self):
        """Test that FolderDiscovery can actually scan Python path for real tools."""
        discovery = FolderDiscovery(scan_python_path=True)

        # Create a temporary directory and add it to sys.path
        with tempfile.TemporaryDirectory() as temp_dir:
            # Store original sys.path and add our temp directory
            original_path = sys.path.copy()
            sys.path.insert(0, temp_dir)

            try:
                # Create a real tool file that matches the pattern *tools*.py
                tool_file = Path(temp_dir) / "real_tools.py"
                tool_file.write_text(
                    '''
from petal.core.decorators import petaltool

@petaltool("real_python_path_tool")
def real_python_path_tool():
    """A real tool found via Python path scanning."""
    return "real result"
'''
                )

                # Actually scan for tools (no mocking of core functionality)
                tools = await discovery._scan_python_path()

                # Verify we found the real tool
                assert "real_python_path_tool" in tools
                assert tools["real_python_path_tool"].name == "real_python_path_tool"

                # Verify it's a real BaseTool instance, not a mock
                from langchain_core.tools import BaseTool

                assert isinstance(tools["real_python_path_tool"], BaseTool)

            finally:
                # Restore original sys.path
                sys.path[:] = original_path

    @pytest.mark.asyncio
    async def test_folder_discovery_parent_directory_exclusion(self):
        """Test that FolderDiscovery excludes files in excluded parent directories."""
        discovery = FolderDiscovery(exclude_patterns=["excluded_dir"])

        # Create a file path that has an excluded parent directory
        file_path = Path("/some/path/excluded_dir/test_tool.py")

        # Should be excluded because parent directory matches pattern
        assert discovery._should_exclude_file(file_path) is True

        # Create a file path with no excluded parents
        file_path = Path("/some/path/valid_dir/test_tool.py")

        # Should not be excluded
        assert discovery._should_exclude_file(file_path) is False

    @pytest.mark.asyncio
    async def test_folder_discovery_handles_module_loading_exceptions(self):
        """Test that FolderDiscovery handles module loading exceptions gracefully."""
        discovery = FolderDiscovery()

        # Create a file that will cause import errors
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
# This will cause a syntax error
invalid syntax here
"""
            )
            f.flush()

            # Should handle the syntax error gracefully
            tools = await discovery._load_tools_from_file(Path(f.name))
            assert tools == {}

            # Clean up
            os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_folder_discovery_scan_python_path_disabled(self):
        """Test that FolderDiscovery respects scan_python_path=False setting."""
        # Create a discovery instance with Python path scanning disabled
        discovery = FolderDiscovery(scan_python_path=False)

        # Store original sys.path
        original_path = sys.path.copy()

        try:
            # Add a temporary directory to sys.path with a tool file
            with tempfile.TemporaryDirectory() as temp_dir:
                sys.path.insert(0, temp_dir)

                # Create a tool file that would be found if scanning was enabled
                tool_file = Path(temp_dir) / "disabled_tools.py"
                tool_file.write_text(
                    '''
from petal.core.decorators import petaltool

@petaltool("disabled_tool")
def disabled_tool():
    """A tool that should not be found when scanning is disabled."""
    return "disabled"
'''
                )

                # Perform full scan with Python path scanning disabled
                await discovery._perform_full_scan()

                # Should not find the tool from Python path when scanning is disabled
                assert "disabled_tool" not in discovery._cached_tools

        finally:
            # Restore original sys.path
            sys.path[:] = original_path

    @pytest.mark.asyncio
    async def test_folder_discovery_recursive_vs_non_recursive(self):
        """Test that FolderDiscovery respects recursive setting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested directory structure
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()

            # Create files at different levels
            root_file = Path(temp_dir) / "root_tool.py"
            sub_file = subdir / "sub_tool.py"

            root_file.write_text(
                '''
from petal.core.decorators import petaltool

@petaltool("root_tool")
def root_tool():
    """Root level tool."""
    return "root"
'''
            )

            sub_file.write_text(
                '''
from petal.core.decorators import petaltool

@petaltool("sub_tool")
def sub_tool():
    """Sub directory tool."""
    return "sub"
'''
            )

            # Test recursive scanning
            discovery_recursive = FolderDiscovery(
                folders=[temp_dir], recursive=True, auto_discover=False
            )

            tools_recursive = await discovery_recursive._scan_folders()

            # Should find both tools (recursive)
            assert "root_tool" in tools_recursive
            assert "sub_tool" in tools_recursive
            assert len(tools_recursive) == 2

            # Test non-recursive scanning
            discovery_non_recursive = FolderDiscovery(
                folders=[temp_dir], recursive=False, auto_discover=False
            )

            tools_non_recursive = await discovery_non_recursive._scan_folders()

            # Should only find root tool (non-recursive)
            assert "root_tool" in tools_non_recursive
            assert "sub_tool" not in tools_non_recursive
            assert len(tools_non_recursive) == 1

    @pytest.mark.asyncio
    async def test_folder_discovery_actual_tool_loading(self):
        """Test that FolderDiscovery can actually load tools from files."""
        discovery = FolderDiscovery()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''
from petal.core.decorators import petaltool

@petaltool("actual_test_tool")
def actual_test_tool():
    """A test tool that should be discovered."""
    return "test result"
'''
            )
            f.flush()

            try:
                # This should actually load the tool
                tools = await discovery._load_tools_from_file(Path(f.name))

                # Should find the tool
                assert "actual_test_tool" in tools
                assert tools["actual_test_tool"].name == "actual_test_tool"

            finally:
                # Clean up
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_folder_discovery_handles_registry_resolution_exceptions(self):
        """Test that FolderDiscovery handles registry resolution exceptions gracefully."""
        discovery = FolderDiscovery()

        # Create a tool file that will register a tool but then fail to resolve
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''
from petal.core.decorators import petaltool

@petaltool("registry_test_tool")
def registry_test_tool():
    """A tool that will be registered but may fail to resolve."""
    return "test"
'''
            )
            f.flush()

            try:
                # This should handle any registry resolution exceptions gracefully
                tools = await discovery._load_tools_from_file(Path(f.name))

                # Should either find the tool or handle the exception gracefully
                # The exact behavior depends on the registry state, but it shouldn't crash
                assert isinstance(tools, dict)

            finally:
                # Clean up
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_folder_discovery_handles_import_errors_gracefully(self):
        """Test that FolderDiscovery handles import errors in tool files gracefully."""
        discovery = FolderDiscovery()

        # Create a file with an import that will fail
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''
from petal.core.decorators import petaltool
from nonexistent_module import some_function  # This will cause an ImportError

@petaltool("import_error_tool")
def import_error_tool():
    """A tool that will fail to import due to missing dependency."""
    return some_function()
'''
            )
            f.flush()

            try:
                # Should handle the import error gracefully
                tools = await discovery._load_tools_from_file(Path(f.name))

                # Should return empty dict when import fails
                assert tools == {}

            finally:
                # Clean up
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_folder_discovery_handles_runtime_errors_in_tool_definition(self):
        """Test that FolderDiscovery handles runtime errors in tool definitions gracefully."""
        discovery = FolderDiscovery()

        # Create a file with a tool that will cause a runtime error when executed
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''
from petal.core.decorators import petaltool

# This will cause a NameError when the module is executed
undefined_variable = some_undefined_function()

@petaltool("runtime_error_tool")
def runtime_error_tool():
    """A tool that will cause a runtime error."""
    return "test"
'''
            )
            f.flush()

            try:
                # Should handle the runtime error gracefully
                tools = await discovery._load_tools_from_file(Path(f.name))

                # Should return empty dict when runtime error occurs
                assert tools == {}

            finally:
                # Clean up
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_scan_python_path_updates_tools_dict(self):
        """Test that _scan_python_path properly updates the tools dictionary."""
        discovery = FolderDiscovery(scan_python_path=True)

        # Store original sys.path
        original_path = sys.path.copy()

        try:
            # Add a temporary directory to sys.path
            with tempfile.TemporaryDirectory() as temp_dir:
                sys.path.insert(0, temp_dir)

                # Create a tool file that matches the pattern
                tool_file = Path(temp_dir) / "dict_update_tools.py"
                tool_file.write_text(
                    '''
from petal.core.decorators import petaltool

@petaltool("dict_update_tool")
def dict_update_tool():
    """A tool to test dictionary updates."""
    return "dict update test"
'''
                )

                # Start with empty tools dict
                # Delete the line: tools = {}

                # Call _scan_python_path and verify it updates the tools dict
                discovered_tools = await discovery._scan_python_path()

                # Should find the tool and update the dict
                assert "dict_update_tool" in discovered_tools
                assert discovered_tools["dict_update_tool"].name == "dict_update_tool"
                assert len(discovered_tools) == 1

        finally:
            # Restore original sys.path
            sys.path[:] = original_path

    @pytest.mark.asyncio
    async def test_load_tools_from_file_registry_diff_mechanism(self):
        """Test that _load_tools_from_file properly handles registry diff mechanism."""
        discovery = FolderDiscovery()

        # Create a tool file that will register a new tool
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''
from petal.core.decorators import petaltool

@petaltool("registry_diff_tool")
def registry_diff_tool():
    """A tool to test registry diff mechanism."""
    return "registry diff test"
'''
            )
            f.flush()

            try:
                # This should trigger the registry diff mechanism
                # The method should detect that a new tool was registered
                tools = await discovery._load_tools_from_file(Path(f.name))

                # Should find the tool through the registry diff mechanism
                assert "registry_diff_tool" in tools
                assert tools["registry_diff_tool"].name == "registry_diff_tool"

            finally:
                # Clean up
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_load_tools_from_file_continues_on_registry_resolve_exception(self):
        """Test that _load_tools_from_file continues processing when registry.resolve() raises an exception."""
        discovery = FolderDiscovery()

        # Create a tool file that defines one BaseTool and registers another tool in the registry
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''
from petal.core.decorators import petaltool
import petal.core.registry

@petaltool("valid_tool")
def valid_tool():
    """A valid tool that will be found in the module namespace."""
    return "valid"

# Register a tool directly in the registry (this will be detected by the diff mechanism)
registry = petal.core.registry.ToolRegistry()
def continue_test_tool():
    """A tool to test continue behavior on registry resolve exception."""
    return "continue test"
registry.register("continue_test_tool", continue_test_tool)
'''
            )
            f.flush()

            try:
                # Mock the registry to simulate a resolve exception for the specific tool
                with patch("petal.core.registry.ToolRegistry") as mock_registry_class:
                    mock_registry = MagicMock()
                    mock_registry_class.return_value = mock_registry

                    # Simulate that a new tool was registered (registry diff)
                    mock_registry.list.side_effect = [
                        {"existing_tool"},
                        {"existing_tool", "continue_test_tool", "valid_tool"},
                    ]

                    # Mock resolve to raise an exception for the new tool, but succeed for the valid tool
                    def mock_resolve(tool_name):
                        if tool_name == "continue_test_tool":
                            raise Exception("Registry resolve failed for this tool")
                        return MagicMock(spec=BaseTool, name=tool_name)

                    mock_registry.resolve.side_effect = mock_resolve

                    # This should handle the registry resolve exception and continue
                    tools = await discovery._load_tools_from_file(Path(f.name))

                    # Should not crash, and should handle the exception gracefully
                    # The tool that failed to resolve should not be in the results
                    assert isinstance(tools, dict)
                    assert "continue_test_tool" not in tools
                    assert "valid_tool" in tools

            finally:
                # Clean up
                os.unlink(f.name)
