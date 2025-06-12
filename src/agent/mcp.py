from .alita_agent import AlitaAgent
from .llm_provider import LLMProvider
from .mcp_box import MCPBox
from .mcp_factory import MCPFactory

# --- Prompts (can be expanded) ---
MULTIMEDIA_RESPONSE_PROMPT = "The tool '{tool_name}' has returned a multimedia response."
SYSTEM_PROMPT = "You are interacting with Alita. How can I assist you?"
NEXT_STEP_PROMPT = "What would you like to do next? (Type 'quit' to exit)"

# Export classes for external use
__all__ = ['AlitaAgent', 'LLMProvider', 'MCPBox', 'MCPFactory', 'SYSTEM_PROMPT', 'MULTIMEDIA_RESPONSE_PROMPT', 'NEXT_STEP_PROMPT']

if __name__ == "__main__":
    print("Initializing Alita Agent (Self-Evolving Demo)...")
    agent = AlitaAgent()

    print(f"\n{SYSTEM_PROMPT}")
    print("Type 'help' for available commands, or 'quit' to exit.")
    print("Try commands like 'add 10 20', 'image a happy cat', or invent new ones like 'greet YourName' or 'circle_area 5'.")

    while True:
        try:
            user_input = input("\nAlita> ").strip()
            if not user_input:
                continue
            
            # Updated to handle streaming response
            full_response_chunks = []
            for chunk in agent.process_command_streaming(user_input):
                if chunk == "quit_signal":
                    print("Exiting Alita agent.")
                    exit() # Exit immediately
                
                # For CLI, print chunks as they arrive and accumulate
                print(chunk, end="", flush=True)
                full_response_chunks.append(str(chunk)) # Store chunk for any post-processing if needed
            
            print() # Newline after all chunks for a command are printed
            # final_response = "".join(full_response_chunks) # If you need the full response string later

        except EOFError: # Handle Ctrl+D
            print("\nExiting Alita agent.")
            break
        except KeyboardInterrupt: # Handle Ctrl+C
            print("\nExiting Alita agent.")
            break

    print("\nAlita Agent session ended.")