from typing import Callable, Dict, List


class Command:
    """
    Encapsulates a callable step as a command object.
    """

    def __init__(self, fn: Callable[[Dict], Dict]):
        self.fn = fn

    def execute(self, state: Dict) -> Dict:
        """
        Execute the command with the given state.

        Args:
            state (Dict): The current agent state.

        Returns:
            Dict: The updated state after this step.
        """
        return self.fn(state)


class Agent:
    """
    The runnable agent object, composed of commands and prompts.
    """

    def __init__(self, commands: List[Command], prompt: str, system_prompt: str):
        self.commands = commands
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.built = True

    def run(self, state: Dict) -> Dict:
        """
        Execute all commands in order, mutating the state.

        Args:
            state (Dict): The initial state.

        Returns:
            Dict: The final state after all steps.
        """
        if not getattr(self, "built", False):
            raise RuntimeError("Agent.run() called before build()")
        for cmd in self.commands:
            state = cmd.execute(state)
        return state


class AgentFactory:
    """
    Builder and fluent interface for constructing Agent objects.
    """

    def __init__(self):
        self._commands: List[Command] = []
        self._prompt_template: str = ""
        self._system_prompt: str = ""
        self._built = False

    def add(self, step: Callable[[Dict], Dict]) -> "AgentFactory":
        """
        Register a step as a Command.

        Args:
            step (Callable[[Dict], Dict]): The step function to add.

        Returns:
            AgentFactory: self (for chaining)
        """
        self._commands.append(Command(step))
        return self

    def with_prompt(self, prompt_template: str) -> "AgentFactory":
        """
        Set the user-visible prompt template.

        Args:
            prompt_template (str): The prompt template string.

        Returns:
            AgentFactory: self (for chaining)
        """
        self._prompt_template = prompt_template
        return self

    def with_system_prompt(self, system_prompt: str) -> "AgentFactory":
        """
        Set the system prompt.

        Args:
            system_prompt (str): The system prompt string.

        Returns:
            AgentFactory: self (for chaining)
        """
        self._system_prompt = system_prompt
        return self

    def build(self) -> Agent:
        """
        Validate and create the Agent.

        Returns:
            Agent: The constructed agent.

        Raises:
            RuntimeError: If no steps have been added.
        """
        if not self._commands:
            raise RuntimeError("Cannot build Agent: no steps added")
        agent = Agent(self._commands, self._prompt_template, self._system_prompt)
        self._built = True
        return agent
