import sys
import os
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import traceback # For detailed error logging

# Adjust path to import AlitaAgent from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # This is src/
src_parent_dir = os.path.dirname(parent_dir) # This is the project root
sys.path.append(src_parent_dir) # Add project root to sys.path

from src.prompt.mcp import AlitaAgent, SYSTEM_PROMPT # Assuming SYSTEM_PROMPT is also in mcp

app = Flask(__name__)

# Initialize agent globally
try:
    print("Initializing Alita Agent for Web UI...")
    agent = AlitaAgent()
    print("Alita Agent initialized successfully for Web UI.")
except Exception as e:
    print(f"CRITICAL: Failed to initialize AlitaAgent: {e}")
    traceback.print_exc()
    agent = None # Ensure agent is defined for checks later

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_command', methods=['POST'])
def send_command():
    if not agent:
        print("Error: /send_command called but AlitaAgent not initialized.")
        return jsonify({'error': "Alita Agent is not available. Initialization failed."}), 503

    data = request.json
    if not data or 'command' not in data:
        return jsonify({'error': 'No command provided or invalid JSON format.'}), 400
    
    command = data['command']
    if not command.strip(): # Check if command is empty or just whitespace
        return jsonify({'error': 'Command cannot be empty.'}), 400

    print(f"Web_App Log: Received command: '{command}'")

    if command.lower() == 'quit':
        return jsonify({'response': "Quit command acknowledged. Client should handle session closure."})

    try:
        def generate_stream():
            print(f"Web_App Log: Calling agent.process_command_streaming for: '{command}'")
            for chunk in agent.process_command_streaming(command):
                print(f"Web_App Log: SERVER YIELDING CHUNK: '{chunk}'") # Added log for each server yield
                yield chunk
            print(f"Web_App Log: Finished streaming for command: '{command}'")
        
        return Response(stream_with_context(generate_stream()), mimetype='text/plain')
    except Exception as e:
        print(f"CRITICAL ERROR in /send_command during stream setup or generation for command '{command}': {e}")
        traceback.print_exc()
        return jsonify({'error': f"A critical server error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    if not agent:
        print("CRITICAL: AlitaAgent failed to initialize. Flask app cannot start.")
    else:
        print("Starting Flask development server for Alita...")
        print("Visit http://127.0.0.1:5001/ in your browser.")
        app.run(debug=True, port=5001)