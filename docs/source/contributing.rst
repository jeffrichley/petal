Contributing to Documentation
=============================

Thank you for your interest in contributing to Petal's documentation! This guide will help you get started.

Building the Documentation
--------------------------

To build the documentation locally:

.. code-block:: bash

   cd docs
   make html

The built documentation will be available in `docs/build/html/`.

Development Setup
-----------------

1. Install development dependencies:

   .. code-block:: bash

      uv pip install -e ".[docs]"

2. Activate the virtual environment:

   .. code-block:: bash

      source .venv/bin/activate

3. Build the documentation:

   .. code-block:: bash

      cd docs && make html

Documentation Structure
-----------------------

The documentation is organized as follows:

- **Getting Started**: Installation and quick start guide
- **API Reference**: Complete API documentation
- **Examples**: Tutorials and code examples
- **Architecture**: Framework design and principles
- **Contributing**: This guide

Writing Guidelines
------------------

When writing documentation:

1. **Use clear, concise language**
2. **Include code examples** for all features
3. **Follow the existing style** and formatting
4. **Test all code examples** before committing
5. **Update the index** when adding new pages

Code Examples
-------------

All code examples should:

- Be complete and runnable
- Follow PEP 8 style guidelines
- Include proper imports
- Show expected output where relevant

.. code-block:: python

   from petal import AgentFactory, tool_fn

   @tool_fn
   def example_tool(input: str) -> str:
       """Example tool function."""
       return f"Processed: {input}"

   agent = AgentFactory().add(example_tool).build()
   result = agent.run({"input": "test"})

API Documentation
-----------------

API documentation is automatically generated from docstrings. Follow these guidelines:

1. **Use Google-style docstrings**:

   .. code-block:: python

      def my_function(param: str) -> str:
          """Brief description.

          Args:
              param: Description of parameter.

          Returns:
              Description of return value.

          Raises:
              ValueError: When param is invalid.
          """

2. **Include type hints** for all parameters and return values
3. **Document all public methods** and classes
4. **Provide usage examples** in docstrings

Building and Testing
--------------------

Before submitting changes:

1. **Build the documentation**:

   .. code-block:: bash

      cd docs && make clean && make html

2. **Check for warnings**:

   .. code-block:: bash

      cd docs && make linkcheck

3. **Test code examples**:

   .. code-block:: bash

      python -m pytest tests/ -v

Submitting Changes
------------------

1. **Create a feature branch** for your changes
2. **Make your changes** following the guidelines above
3. **Test your changes** by building the documentation
4. **Submit a pull request** with a clear description

Thank you for contributing to Petal's documentation! 