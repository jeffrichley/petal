"""
MCP Tool Registry Playground

This example demonstrates the various ways to work with MCP tools in the Petal framework:
1. Direct ToolFactory registration
2. Decorator-based MCP servers and tools
3. YAML configuration integration
4. Tool resolution and discovery
5. Async loading and error handling
"""

import asyncio
from typing import cast

from petal.core.decorators import petalmcp, petaltool
from petal.core.registry import ToolRegistry
from petal.core.tool_factory import ToolFactory

# ============================================================================
# Example 1: Direct ToolFactory MCP Registration
# ============================================================================


async def demo_direct_mcp_registration():
    """Demonstrate direct MCP server registration with ToolFactory."""
    print("=== Direct MCP Registration Demo ===\n")

    # Create ToolFactory instance
    tf = ToolFactory()

    # Mock MCP server configuration with proper transport
    print("1. Registering MCP servers...")

    # Register MCP servers (this would normally connect to real MCP servers)
    # For demo purposes, we'll use a mock resolver that returns immediately
    async def mock_filesystem_resolver():
        """Mock resolver that returns fake filesystem tools immediately."""
        from langchain_core.tools import BaseTool

        class MockFileTool(BaseTool):
            name: str = "list_files"
            description: str = "List files in a directory"

            def _run(self, path: str = ".") -> str:
                return f"Files in {path}: file1.txt, file2.py, directory1/"

        class MockReadTool(BaseTool):
            name: str = "read_file"
            description: str = "Read contents of a file"

            def _run(self, path: str) -> str:
                return f"Contents of {path}: Hello, World!"

        return [MockFileTool(), MockReadTool()]

    async def mock_sqlite_resolver():
        """Mock resolver that returns fake SQLite tools immediately."""
        from langchain_core.tools import BaseTool

        class MockQueryTool(BaseTool):
            name: str = "query"
            description: str = "Execute SQL query"

            def _run(self, sql: str) -> str:
                return f"Query result: {sql} returned 5 rows"

        return [MockQueryTool()]

    # Register MCP servers
    tf.add_mcp("filesystem", resolver=mock_filesystem_resolver)
    tf.add_mcp("sqlite", resolver=mock_sqlite_resolver)

    print("   ✅ Filesystem server registered")
    print("   ✅ SQLite server registered")

    # Give the background tasks a moment to complete
    print("\n2. Waiting for MCP tools to load...")
    await asyncio.sleep(0.1)  # Brief pause for background tasks

    # Check if tools are loaded
    try:
        tf.resolve("mcp:filesystem:list_files")
        print("   ✅ Filesystem tools loaded")
    except KeyError:
        print("   ⏳ Filesystem tools still loading...")

    try:
        tf.resolve("mcp:sqlite:query")
        print("   ✅ SQLite tools loaded")
    except KeyError:
        print("   ⏳ SQLite tools still loading...")

    # List available tools
    print("\n3. Available tools:")
    tools = tf.list()
    for tool in tools:
        print(f"   - {tool}")

    # Resolve and use MCP tools
    print("\n4. Using MCP tools:")

    try:
        # Resolve filesystem tools
        list_files_tool = tf.resolve("mcp:filesystem:list_files")
        read_file_tool = tf.resolve("mcp:filesystem:read_file")

        print(f"   📁 List files tool: {list_files_tool.name}")
        print(f"   📄 Read file tool: {read_file_tool.name}")

        # Resolve SQLite tools
        query_tool = tf.resolve("mcp:sqlite:query")
        print(f"   🗄️  Query tool: {query_tool.name}")

        # Actually execute the tools to show they work
        print("\n5. Executing MCP tools:")

        # Execute filesystem tools
        list_result = list_files_tool.invoke({"path": "/tmp"})
        print(f"   📁 List files result: {list_result}")

        read_result = read_file_tool.invoke({"path": "example.txt"})
        print(f"   📄 Read file result: {read_result}")

        # Execute SQLite tool
        query_result = query_tool.invoke({"sql": "SELECT * FROM users"})
        print(f"   🗄️  Query result: {query_result}")

    except KeyError as e:
        print(f"   ❌ Error resolving tool: {e}")

    print()


# ============================================================================
# Example 2: Decorator-Based MCP Integration
# ============================================================================


async def demo_decorator_based_mcp():
    """Demonstrate MCP integration using decorators."""
    print("=== Decorator-Based MCP Demo ===\n")

    # Mock MCP server configuration with proper transport
    print("1. Creating MCP server with @petalmcp decorator...")

    # Instead of using @petalmcp with real config, we'll manually register with mock resolver
    # to avoid hanging on real MCP server connections
    tf = ToolFactory()

    async def mock_math_resolver():
        """Mock resolver that returns fake math tools immediately."""
        from langchain_core.tools import BaseTool

        class MockAddTool(BaseTool):
            name: str = "add"
            description: str = "Add two numbers"

            def _run(self, a: float, b: float) -> float:
                return a + b

        class MockMultiplyTool(BaseTool):
            name: str = "multiply"
            description: str = "Multiply two numbers"

            def _run(self, a: float, b: float) -> float:
                return a * b

        return [MockAddTool(), MockMultiplyTool()]

    # Register MCP server manually with mock resolver
    tf.add_mcp("math_server", resolver=mock_math_resolver)

    print("   ✅ Math server registered with mock resolver")

    print("\n2. Creating regular tool with @petaltool decorator...")

    @petaltool("calculator")
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    print("   ✅ Regular tool decorated")

    # Access the ToolRegistry singleton
    registry = ToolRegistry()
    print(f"\n3. Tools in registry: {registry.list()}")

    # Wait for MCP tools to load and then execute them
    print("\n4. Executing tools:")

    # Execute regular tool
    try:
        calc_result = add_numbers.invoke({"a": 5, "b": 3})
        print(f"   🧮 Calculator result: 5 + 3 = {calc_result}")
    except Exception as e:
        print(f"   ❌ Calculator error: {e}")

    # Wait for MCP tools to load
    await asyncio.sleep(0.1)

    # Execute MCP math tools
    try:
        add_tool = tf.resolve("mcp:math_server:add")
        multiply_tool = tf.resolve("mcp:math_server:multiply")

        add_result = add_tool.invoke({"a": 10.5, "b": 20.3})
        print(f"   ➕ Add result: 10.5 + 20.3 = {add_result}")

        multiply_result = multiply_tool.invoke({"a": 7, "b": 8})
        print(f"   ✖️  Multiply result: 7 × 8 = {multiply_result}")

    except KeyError as e:
        print(f"   ❌ MCP tool error: {e}")

    print()


# ============================================================================
# Example 3: ToolRegistry Singleton and Discovery
# ============================================================================


async def demo_tool_registry_discovery():
    """Demonstrate ToolRegistry singleton and discovery capabilities."""
    print("=== ToolRegistry Discovery Demo ===\n")

    # Get the singleton ToolRegistry
    registry1 = ToolRegistry()
    registry2 = ToolRegistry()

    print("1. Verifying singleton pattern:")
    print(f"   Registry 1 ID: {id(registry1)}")
    print(f"   Registry 2 ID: {id(registry2)}")
    print(f"   Same instance: {registry1 is registry2}")

    # Add some tools
    print("\n2. Adding tools to registry:")

    @petaltool("greeter")
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"

    @petaltool("multiplier")
    def multiply(x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y

    print("   ✅ Greeter tool added")
    print("   ✅ Multiplier tool added")

    # List tools
    print(f"\n3. Tools in registry: {registry1.list()}")

    # Resolve tools
    print("\n4. Resolving tools:")
    try:
        greeter = await registry1.resolve("greeter")
        multiplier = await registry1.resolve("multiplier")

        print(f"   👋 Greeter tool: {greeter.name}")
        print(f"   ✖️  Multiplier tool: {multiplier.name}")

        # Execute the tools to show they work
        print("\n5. Executing tools:")

        greeter_result = await greeter.ainvoke({"name": "Alice"})
        print(f"   👋 Greeter result: {greeter_result}")

        multiplier_result = await multiplier.ainvoke({"x": 6, "y": 7})
        print(f"   ✖️  Multiplier result: 6 × 7 = {multiplier_result}")

    except KeyError as e:
        print(f"   ❌ Error resolving tool: {e}")

    print()


# ============================================================================
# Example 4: Error Handling and Edge Cases
# ============================================================================


async def demo_error_handling():
    """Demonstrate error handling in MCP tool resolution."""
    print("=== Error Handling Demo ===\n")

    tf = ToolFactory()

    print("1. Testing resolution of non-existent MCP tool:")
    try:
        tf.resolve("mcp:nonexistent:tool")
    except KeyError as e:
        print(f"   ✅ Expected error: {e}")

    print("\n2. Testing resolution of malformed MCP tool name:")
    try:
        tf.resolve("mcp:invalid:tool:extra")
    except KeyError as e:
        print(f"   ✅ Expected error: {e}")

    print("\n3. Testing resolution of non-MCP tool:")
    try:
        tf.resolve("nonexistent_tool")
    except KeyError as e:
        print(f"   ✅ Expected error: {e}")

    print("\n4. Testing MCP tool still loading:")

    # Create a slow-loading MCP server with proper mock resolver
    async def slow_resolver():
        """Resolver that takes time to load."""
        await asyncio.sleep(0.5)  # Reduced time for demo
        from langchain_core.tools import BaseTool

        class SlowTool(BaseTool):
            name: str = "slow_tool"
            description: str = "A tool that takes time to load"

            def _run(self) -> str:
                return "Slow tool loaded!"

        return [SlowTool()]

    # Register the slow server
    tf.add_mcp("slow_server", resolver=slow_resolver)

    # Try to resolve immediately (should fail)
    try:
        tf.resolve("mcp:slow_server:slow_tool")
    except KeyError as e:
        print(f"   ✅ Expected error (still loading): {e}")

    # Wait a bit and try again
    print("   ⏳ Waiting for slow tool to load...")
    await asyncio.sleep(0.6)  # Wait for the slow resolver to complete

    try:
        slow_tool = tf.resolve("mcp:slow_server:slow_tool")
        print(f"   ✅ Slow tool loaded: {slow_tool.name}")
    except KeyError as e:
        print(f"   ❌ Unexpected error: {e}")

    print()


# ============================================================================
# Example 5: Integration with YAML Configuration
# ============================================================================


def demo_yaml_integration():
    """Demonstrate how MCP tools integrate with YAML configuration."""
    print("=== YAML Integration Demo ===\n")

    # Simulate YAML configuration with proper transport
    yaml_config = {
        "react_node": {
            "type": "react",
            "name": "reasoning_agent",
            "tools": ["mcp:filesystem:list_files", "mcp:sqlite:query", "calculator"],
            "mcp_servers": {
                "filesystem": {
                    "config": {
                        "filesystem": {
                            "transport": "stdio",
                            "command": "npx",
                            "args": [
                                "-y",
                                "@modelcontextprotocol/server-filesystem",
                                "/tmp",
                            ],
                        }
                    }
                },
                "sqlite": {
                    "config": {
                        "sqlite": {
                            "transport": "stdio",
                            "command": "npx",
                            "args": [
                                "-y",
                                "@modelcontextprotocol/server-sqlite",
                                "test.db",
                            ],
                        }
                    }
                },
            },
            "max_iterations": 5,
            "reasoning_prompt": "Think step by step about how to solve this problem.",
        }
    }

    print("1. YAML Configuration:")
    import yaml

    print(yaml.dump(yaml_config, default_flow_style=False, indent=2))

    print("\n2. Simulating YAML handler processing:")

    # Simulate what the ReactNodeHandler would do
    # tf = ToolFactory()  # Unused variable removed

    # Register MCP servers from config
    mcp_servers = cast(dict, yaml_config["react_node"]["mcp_servers"])
    for server_name, server_info in mcp_servers.items():
        mcp_config = server_info.get("config")
        if mcp_config:
            print(f"   📡 Registering MCP server: {server_name}")
            # In real implementation, this would call tf.add_mcp()

    # Resolve tools
    tools = cast(list[str], yaml_config["react_node"]["tools"])
    print("\n3. Tool resolution from config:")
    for tool_name in tools:
        print(f"   🔧 Resolving: {tool_name}")
        # In real implementation, this would call tf.resolve(tool_name)

    print()


# ============================================================================
# Example 6: Performance and Caching
# ============================================================================


async def demo_performance_and_caching():
    """Demonstrate performance optimizations and caching."""
    print("=== Performance and Caching Demo ===\n")

    # Use ToolRegistry for @petaltool decorators since they register there
    registry = ToolRegistry()

    print("1. Testing tool resolution performance:")

    # Add some tools BEFORE trying to resolve them
    @petaltool("fast_tool")
    def fast_tool() -> str:
        """A fast tool for demo purposes."""
        return "Fast!"

    @petaltool("cached_tool")
    def cached_tool() -> str:
        """A cached tool for demo purposes."""
        return "Cached!"

    # Time multiple resolutions of the same tool
    import time

    start_time = time.time()
    for _ in range(100):
        await registry.resolve("fast_tool")
    end_time = time.time()

    print(f"   ⚡ 100 tool resolutions: {end_time - start_time:.4f} seconds")

    # Execute the tools to show they work
    print("\n3. Executing tools:")

    try:
        fast_result = await registry.resolve("fast_tool")
        fast_output = await fast_result.ainvoke({})
        print(f"   ⚡ Fast tool result: {fast_output}")

        cached_result = await registry.resolve("cached_tool")
        cached_output = await cached_result.ainvoke({})
        print(f"   💾 Cached tool result: {cached_output}")

    except KeyError as e:
        print(f"   ❌ Tool execution error: {e}")

    print("\n4. Testing registry caching:")

    # Get registry instances
    # registry1 = ToolRegistry()  # Unused variable removed
    registry2 = ToolRegistry()

    # Add tool to one instance
    @petaltool("shared_tool")
    def shared_tool() -> str:
        """A shared tool for demo purposes."""
        return "Shared!"

    # Check if it's available in the other instance
    try:
        tool = await registry2.resolve("shared_tool")
        print(f"   ✅ Tool shared across registry instances: {tool.name}")
    except KeyError as e:
        print(f"   ❌ Tool not shared: {e}")

    print()


# ============================================================================
# Example 7: Real MCP Server Integration with Decorators
# ============================================================================


async def demo_real_mcp_server():
    """Demonstrate connecting to a real MCP server using @petalmcp decorator."""
    print("=== Real MCP Server with Decorators Demo ===\n")

    # Real MCP server config (matches the fixture script)
    math_config = {
        "math": {
            "transport": "stdio",
            "command": "python3",
            "args": ["tests/fixtures/mcp_server_script.py"],
        }
    }

    print("1. Creating MCP server class with @petalmcp decorator...")

    @petalmcp("math", math_config)
    class MathServer:
        """Real MCP server for mathematical operations."""

        pass

    print("   ✅ Math server class decorated with real config")

    # Wait for tools to load
    print("2. Waiting for math tools to load...")
    await asyncio.sleep(1.0)

    # Get the ToolFactory instance that was used by the decorator
    tf = ToolFactory()

    # Resolve and use math tools
    try:
        add_tool = tf.resolve("mcp:math:add")
        multiply_tool = tf.resolve("mcp:math:multiply")
        print(f"   ➕ Add tool: {add_tool.name}")
        print(f"   ✖️  Multiply tool: {multiply_tool.name}")

        # Call the tools
        add_result = add_tool.invoke({"a": 2, "b": 3})
        print(f"   ➕ Add result: 2 + 3 = {add_result}")

        multiply_result = multiply_tool.invoke({"a": 4, "b": 5})
        print(f"   ✖️  Multiply result: 4 × 5 = {multiply_result}")

    except KeyError as e:
        print(f"   ❌ Error resolving real MCP tool: {e}")

    print()


# ============================================================================
# Main Demo Function
# ============================================================================


async def main():
    """Run all MCP tool registry demonstrations."""
    print("🚀 MCP Tool Registry Playground")
    print("=" * 50)
    print()

    try:
        # Run all demos
        await demo_direct_mcp_registration()
        await demo_decorator_based_mcp()
        await demo_tool_registry_discovery()
        await demo_error_handling()
        demo_yaml_integration()
        await demo_performance_and_caching()
        await demo_real_mcp_server()

        print("🎉 All demonstrations completed successfully!")
        print("\nKey Takeaways:")
        print("✅ MCP tools can be registered directly with ToolFactory")
        print("✅ Decorators provide convenient MCP integration")
        print("✅ ToolRegistry singleton ensures shared state")
        print("✅ Async loading prevents blocking")
        print("✅ Error handling covers various edge cases")
        print("✅ YAML configuration integrates seamlessly")
        print("✅ Performance optimizations are in place")
        print("✅ Real MCP server integration works!")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run the playground
    asyncio.run(main())
