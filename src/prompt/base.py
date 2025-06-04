class BasePrompt:
    """Base class for prompts used in agents."""
    
    def __init__(self, prompt_template: str):
        self.prompt_template = prompt_template

    def format_prompt(self, **kwargs) -> str:
        """Format the prompt with the given keyword arguments."""
        return self.prompt_template.format(**kwargs)

    def get_template(self) -> str:
        """Return the prompt template."""
        return self.prompt_template

# Additional utility functions for prompt handling can be added here.