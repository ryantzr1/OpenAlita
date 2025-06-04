class BaseTool:
    """Base class for tools that agents can use."""

    def __init__(self, name: str):
        self.name = name

    def execute(self, *args, **kwargs):
        """Execute the tool with the given arguments."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_info(self) -> dict:
        """Return information about the tool."""
        return {
            "name": self.name,
            "description": self.__doc__,
        }