class BaseAgent:
    """Base class for all agents in the Open-Alita framework."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def initialize(self, **kwargs) -> None:
        """Initialize the agent with the given parameters."""
        raise NotImplementedError("Subclasses should implement this method.")

    def think(self) -> bool:
        """Process the current state and decide the next action."""
        raise NotImplementedError("Subclasses should implement this method.")

    def cleanup(self) -> None:
        """Clean up resources when the agent is done."""
        raise NotImplementedError("Subclasses should implement this method.")