from unittest.mock import Mock

import pytest
from tests.petal.conftest_factory import ChatState
from typing_extensions import TypedDict

from petal.core.agent import Agent
from petal.core.factory import AgentFactory


class TestDiagramAgent:
    """Test the diagram_agent static method."""

    def test_diagram_agent_success_png(self, tmp_path):
        """Test successful diagram generation in PNG format."""
        from langgraph.graph import END, START, StateGraph

        class TestState(TypedDict):
            test: str

        # Create a simple graph
        graph = StateGraph(TestState)
        graph.add_node("test", lambda x: x)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)
        compiled_graph = graph.compile()

        # Create and build agent
        agent = Agent().build(compiled_graph, TestState)

        # Mock the graph object to return a mock with draw_mermaid_png
        mock_graph_obj = Mock()
        mock_graph_obj.draw_mermaid_png = Mock(return_value=b"fake_png_data")
        # Use setattr to properly mock the method
        assert agent.graph is not None  # type: ignore[unreachable]
        agent.graph.get_graph = Mock(return_value=mock_graph_obj)

        # Test diagram generation
        output_path = tmp_path / "test_diagram.png"
        AgentFactory.diagram_agent(agent, str(output_path), "png")

        # Verify the mock was called correctly
        mock_graph_obj.draw_mermaid_png.assert_called_once()
        assert output_path.exists()

    def test_diagram_agent_success_svg(self, tmp_path):
        """Test successful diagram generation in SVG format."""
        from langgraph.graph import END, START, StateGraph

        class TestState(TypedDict):
            test: str

        # Create a simple graph
        graph = StateGraph(TestState)
        graph.add_node("test", lambda x: x)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)
        compiled_graph = graph.compile()

        # Create and build agent
        agent = Agent().build(compiled_graph, TestState)

        # Mock the graph object to return a mock with draw_mermaid_svg
        mock_graph_obj = Mock()
        mock_graph_obj.draw_mermaid_svg = Mock(return_value=b"fake_svg_data")
        # Use setattr to properly mock the method
        assert agent.graph is not None  # type: ignore[unreachable]
        agent.graph.get_graph = Mock(return_value=mock_graph_obj)

        # Test diagram generation
        output_path = tmp_path / "test_diagram.svg"
        AgentFactory.diagram_agent(agent, str(output_path), "svg")

        # Verify the mock was called correctly
        mock_graph_obj.draw_mermaid_svg.assert_called_once()
        assert output_path.exists()

    def test_diagram_agent_agent_not_built(self):
        """Test that diagram_agent raises error when agent is not built."""

        # Create agent without building it
        agent = Agent()
        agent.built = False

        with pytest.raises(
            RuntimeError, match="Agent must be built before generating diagram"
        ):
            AgentFactory.diagram_agent(agent, "test.png")

    def test_diagram_agent_graph_is_none(self):
        """Test that diagram_agent raises error when agent.graph is None."""

        # Create agent with built=True but graph=None
        agent = Agent()
        agent.built = True
        agent.graph = None

        with pytest.raises(
            RuntimeError, match="Agent must be built before generating diagram"
        ):
            AgentFactory.diagram_agent(agent, "test.png")

    def test_diagram_agent_unsupported_format(self):
        """Test that diagram_agent raises error for unsupported formats."""
        from langgraph.graph import END, START, StateGraph

        class TestState(TypedDict):
            test: str

        # Create a simple graph
        graph = StateGraph(TestState)
        graph.add_node("test", lambda x: x)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)
        compiled_graph = graph.compile()

        # Create and build agent
        agent = Agent().build(compiled_graph, TestState)

        # Mock the graph object
        mock_graph_obj = Mock()
        # Use setattr to properly mock the method
        assert agent.graph is not None  # type: ignore[unreachable]
        agent.graph.get_graph = Mock(return_value=mock_graph_obj)

        with pytest.raises(RuntimeError, match="Unsupported format: pdf"):
            AgentFactory.diagram_agent(agent, "test.pdf", "pdf")

    def test_diagram_agent_graph_no_mermaid_support(self):
        """Test that diagram_agent raises error when graph doesn't support mermaid."""
        from langgraph.graph import END, START, StateGraph

        class TestState(TypedDict):
            test: str

        # Create a simple graph
        graph = StateGraph(TestState)
        graph.add_node("test", lambda x: x)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)
        compiled_graph = graph.compile()

        # Create and build agent
        agent = Agent().build(compiled_graph, TestState)

        # Create a simple object without mermaid methods
        simple_graph_obj = object()
        # Use setattr to properly mock the method
        assert agent.graph is not None  # type: ignore[unreachable]
        agent.graph.get_graph = Mock(return_value=simple_graph_obj)

        with pytest.raises(
            RuntimeError,
            match="Graph object doesn't support mermaid diagram generation",
        ):
            AgentFactory.diagram_agent(agent, "test.png")

    def test_diagram_agent_file_write_error(self, tmp_path):
        """Test that diagram_agent handles file write errors gracefully."""
        from langgraph.graph import END, START, StateGraph

        class TestState(TypedDict):
            test: str

        # Create a simple graph
        graph = StateGraph(TestState)
        graph.add_node("test", lambda x: x)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)
        compiled_graph = graph.compile()

        # Create and build agent
        agent = Agent().build(compiled_graph, TestState)

        # Mock the graph object
        mock_graph_obj = Mock()
        mock_graph_obj.draw_mermaid_png = Mock(return_value=b"fake_png_data")
        # Use setattr to properly mock the method
        assert agent.graph is not None  # type: ignore[unreachable]
        agent.graph.get_graph = Mock(return_value=mock_graph_obj)

        # Test with a path that can't be written to (directory instead of file)
        directory_path = tmp_path / "test_dir"
        directory_path.mkdir()

        with pytest.raises(RuntimeError, match="Failed to generate diagram"):
            AgentFactory.diagram_agent(agent, str(directory_path), "png")

    def test_diagram_agent_graph_get_graph_error(self):
        """Test that diagram_agent handles graph.get_graph() errors gracefully."""
        from langgraph.graph import END, START, StateGraph

        class TestState(TypedDict):
            test: str

        # Create a simple graph
        graph = StateGraph(TestState)
        graph.add_node("test", lambda x: x)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)
        compiled_graph = graph.compile()

        # Create and build agent
        agent = Agent().build(compiled_graph, TestState)

        # Mock get_graph to raise an exception
        # Use setattr to properly mock the method
        assert agent.graph is not None  # type: ignore[unreachable]
        agent.graph.get_graph = Mock(side_effect=Exception("Graph error"))

        with pytest.raises(
            RuntimeError, match="Failed to generate diagram: Graph error"
        ):
            AgentFactory.diagram_agent(agent, "test.png")

    def test_diagram_agent_mermaid_method_error(self):
        """Test that diagram_agent handles mermaid method errors gracefully."""
        from langgraph.graph import END, START, StateGraph

        class TestState(TypedDict):
            test: str

        # Create a simple graph
        graph = StateGraph(TestState)
        graph.add_node("test", lambda x: x)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)
        compiled_graph = graph.compile()

        # Create and build agent
        agent = Agent().build(compiled_graph, TestState)

        # Mock the graph object with a method that raises an exception
        mock_graph_obj = Mock()
        mock_graph_obj.draw_mermaid_png = Mock(side_effect=Exception("Mermaid error"))
        # Use setattr to properly mock the method
        assert agent.graph is not None  # type: ignore[unreachable]
        agent.graph.get_graph = Mock(return_value=mock_graph_obj)

        with pytest.raises(
            RuntimeError, match="Failed to generate diagram: Mermaid error"
        ):
            AgentFactory.diagram_agent(agent, "test.png")


class TestDiagramGraph:
    """Test the diagram_graph method."""

    @pytest.mark.asyncio
    async def test_diagram_graph_success(self, tmp_path):
        """Test successful diagram generation with diagram_graph."""
        from langchain_core.tools import tool

        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Test: {query}"

        # Create factory with steps
        factory = AgentFactory(ChatState)
        factory.with_chat().with_tools([test_tool])

        # Mock the build method to return a working agent
        mock_agent = Mock()
        mock_agent.built = True
        mock_agent.graph = Mock()

        # Mock the graph object
        mock_graph_obj = Mock()
        mock_graph_obj.draw_mermaid_png = Mock(return_value=b"fake_png_data")
        # Use setattr to properly mock the method
        assert mock_agent.graph is not None  # type: ignore[unreachable]
        mock_agent.graph.get_graph = Mock(return_value=mock_graph_obj)

        # Mock build to return an awaitable
        build_called = False

        async def mock_build():
            nonlocal build_called
            build_called = True
            return mock_agent

        factory.build = mock_build  # type: ignore[method-assign]

        # Test diagram generation
        output_path = tmp_path / "test_diagram.png"
        await factory.diagram_graph(str(output_path), "png")

        # Verify the mocks were called correctly
        assert build_called
        mock_graph_obj.draw_mermaid_png.assert_called_once()
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_diagram_graph_no_steps_raises_error(self):
        """Test that diagram_graph raises error when no steps are configured."""
        factory = AgentFactory(ChatState)
        # Don't add any steps

        with pytest.raises(
            ValueError, match="Cannot generate diagram: no steps have been configured"
        ):
            await factory.diagram_graph("test.png")

    @pytest.mark.asyncio
    async def test_diagram_graph_build_failure_propagates(self):
        """Test that diagram_graph propagates build failures."""
        from langchain_core.tools import tool

        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Test: {query}"

        # Create factory with steps
        factory = AgentFactory(ChatState)
        factory.with_chat().with_tools([test_tool])

        # Mock build to raise an exception
        async def mock_build_fail():
            raise Exception("Build failed")

        factory.build = mock_build_fail  # type: ignore[method-assign]

        with pytest.raises(Exception, match="Build failed"):
            await factory.diagram_graph("test.png")

    @pytest.mark.asyncio
    async def test_diagram_graph_diagram_agent_failure_propagates(self):
        """Test that diagram_graph propagates diagram_agent failures."""
        from langchain_core.tools import tool

        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Test: {query}"

        # Create factory with steps
        factory = AgentFactory(ChatState)
        factory.with_chat().with_tools([test_tool])

        # Mock the build method to return an agent that will fail diagram generation
        mock_agent = Mock()
        mock_agent.built = True
        mock_agent.graph = Mock()

        # Mock get_graph to raise an exception
        # Use setattr to properly mock the method
        assert mock_agent.graph is not None  # type: ignore[unreachable]
        mock_agent.graph.get_graph = Mock(side_effect=Exception("Graph error"))

        # Mock build to return an awaitable
        async def mock_build():
            return mock_agent

        factory.build = mock_build  # type: ignore[method-assign]

        with pytest.raises(
            RuntimeError, match="Failed to generate diagram: Graph error"
        ):
            await factory.diagram_graph("test.png")

    @pytest.mark.asyncio
    async def test_diagram_graph_with_different_formats(self, tmp_path):
        """Test diagram_graph with different output formats."""
        from langchain_core.tools import tool

        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Test: {query}"

        # Create factory with steps
        factory = AgentFactory(ChatState)
        factory.with_chat().with_tools([test_tool])

        # Mock the build method to return a working agent
        mock_agent = Mock()
        mock_agent.built = True
        mock_agent.graph = Mock()

        # Mock the graph object for SVG
        mock_graph_obj = Mock()
        mock_graph_obj.draw_mermaid_svg = Mock(return_value=b"fake_svg_data")
        # Use setattr to properly mock the method
        assert mock_agent.graph is not None  # type: ignore[unreachable]
        mock_agent.graph.get_graph = Mock(return_value=mock_graph_obj)

        # Mock build to return an awaitable
        build_called = False

        async def mock_build():
            nonlocal build_called
            build_called = True
            return mock_agent

        factory.build = mock_build  # type: ignore[method-assign]

        # Test SVG diagram generation
        output_path = tmp_path / "test_diagram.svg"
        await factory.diagram_graph(str(output_path), "svg")

        # Verify the mocks were called correctly
        assert build_called
        mock_graph_obj.draw_mermaid_svg.assert_called_once()
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_diagram_graph_returns_none(self, tmp_path):
        """Test that diagram_graph returns None (not self for fluent chaining)."""
        from langchain_core.tools import tool

        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Test: {query}"

        # Create factory with steps
        factory = AgentFactory(ChatState)
        factory.with_chat().with_tools([test_tool])

        # Mock the build method to return a working agent
        mock_agent = Mock()
        mock_agent.built = True
        mock_agent.graph = Mock()

        # Mock the graph object
        mock_graph_obj = Mock()
        mock_graph_obj.draw_mermaid_png = Mock(return_value=b"fake_png_data")
        # Use setattr to properly mock the method
        assert mock_agent.graph is not None  # type: ignore[unreachable]
        mock_agent.graph.get_graph = Mock(return_value=mock_graph_obj)

        # Mock build to return an awaitable
        async def mock_build():
            return mock_agent

        factory.build = mock_build  # type: ignore[method-assign]

        # Test that diagram_graph returns None
        output_path = tmp_path / "test_diagram.png"
        result = await factory.diagram_graph(str(output_path), "png")

        # Verify it returns None (not self for fluent chaining)
        assert result is None
        assert output_path.exists()
