from .alita_agent import AlitaAgent
from .llm_provider import LLMProvider
from .mcp_box import MCPBox
from .mcp_factory import MCPFactory
import logging
import os
import sys
import time
from datetime import datetime

# --- Setup Logging ---
def setup_logging():
    """Setup logging configuration for Alita"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'alita_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('alita')

# --- Prompts (can be expanded) ---
MULTIMEDIA_RESPONSE_PROMPT = "The tool '{tool_name}' has returned a multimedia response."
SYSTEM_PROMPT = "You are interacting with Alita. How can I assist you?"
NEXT_STEP_PROMPT = "What would you like to do next? (Type 'quit' to exit)"

# Export classes for external use
__all__ = ['AlitaAgent', 'LLMProvider', 'MCPBox', 'MCPFactory', 'SYSTEM_PROMPT', 'MULTIMEDIA_RESPONSE_PROMPT', 'NEXT_STEP_PROMPT']

if __name__ == "__main__":
    print("Initializing Alita Agent (Self-Evolving Demo)...")
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Alita Agent")
    
    # Initialize agent
    try:
        agent = AlitaAgent()
        logger.info("Alita Agent successfully initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Alita Agent: {e}", exc_info=True)
        print(f"Error: Failed to initialize Alita Agent: {e}")
        sys.exit(1)

    print(f"\n{SYSTEM_PROMPT}")
    print("Type 'help' for available commands, 'log' to view logs, or 'quit' to exit.")
    print("Try commands like 'add 10 20', 'image a happy cat', or invent new ones like 'greet YourName' or 'circle_area 5'.")
    
    logger.info("Alita Agent ready for user input")

    while True:
        try:
            user_input = input("\nAlita> ").strip()
            if not user_input:
                continue
            
            # Special command to view logs
            if user_input.lower() == 'log':
                print("\n--- Showing the most recent log entries ---")
                try:
                    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
                    log_files = sorted([f for f in os.listdir(log_dir) if f.startswith('alita_')], reverse=True)
                    
                    if not log_files:
                        print("No log files found.")
                    else:
                        latest_log = os.path.join(log_dir, log_files[0])
                        with open(latest_log, 'r') as f:
                            lines = f.readlines()
                            # Show the last 20 lines or fewer if there are fewer lines
                            for line in lines[-min(20, len(lines)):]:
                                print(line.strip())
                except Exception as e:
                    print(f"Error reading logs: {e}")
                continue
                
            logger.info(f"User input: {user_input}")
            
            # Updated to handle streaming response
            full_response_chunks = []
            try:
                for chunk in agent.process_command_streaming(user_input):
                    if chunk == "quit_signal":
                        logger.info("User requested to exit")
                        print("Exiting Alita agent.")
                        sys.exit(0)  # Exit immediately
                    
                    # For CLI, print chunks as they arrive and accumulate
                    print(chunk, end="", flush=True)
                    full_response_chunks.append(str(chunk))  # Store chunk for any post-processing if needed
                
                print()  # Newline after all chunks for a command are printed
                logger.info(f"Command executed successfully: {user_input}")
            except Exception as e:
                logger.error(f"Error processing command '{user_input}': {e}", exc_info=True)
                print(f"\nError: {e}")

        except EOFError:  # Handle Ctrl+D
            logger.info("Received EOF, shutting down")
            print("\nExiting Alita agent.")
            break
        except KeyboardInterrupt:  # Handle Ctrl+C
            logger.info("Received keyboard interrupt, shutting down")
            print("\nExiting Alita agent.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            print(f"\nUnexpected error: {e}")

    logger.info("Alita Agent session ended")
    print("\nAlita Agent session ended.")