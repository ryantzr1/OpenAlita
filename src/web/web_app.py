import sys
import os
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_parent_dir = os.path.dirname(parent_dir)
sys.path.append(src_parent_dir)

from src.agent.mcp import AlitaAgent, SYSTEM_PROMPT

app = Flask(__name__)

try:
    print("Initializing Alita Agent for Web UI...")
    agent = AlitaAgent()
    print("Alita Agent initialized successfully for Web UI.")
except Exception as e:
    print(f"CRITICAL: Failed to initialize AlitaAgent: {e}")
    traceback.print_exc()
    agent = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_command', methods=['POST'])
def send_command():
    if not agent:
        return jsonify({'error': "Alita Agent is not available. Initialization failed."}), 503

    data = request.json
    if not data or 'command' not in data:
        return jsonify({'error': 'No command provided or invalid JSON format.'}), 400
    
    command = data['command']
    if not command.strip():
        return jsonify({'error': 'Command cannot be empty.'}), 400

    if command.lower() == 'quit':
        return jsonify({'response': "Quit command acknowledged. Client should handle session closure."})

    try:
        def generate_stream():
            for chunk in agent.process_command_streaming(command):
                yield chunk
        
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