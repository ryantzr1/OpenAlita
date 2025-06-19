import sys
import os
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import traceback
import asyncio
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_parent_dir = os.path.dirname(parent_dir)
sys.path.append(src_parent_dir)

from src.agent.alita_agent import AlitaAgent

app = Flask(__name__)

try:
    print("Initializing Enhanced Alita Agent with LangGraph for Web UI...")
    agent = AlitaAgent()
    print("✅ Enhanced Alita Agent with LangGraph initialized successfully for Web UI.")
    
    # Check LangGraph status
    status = agent.get_workflow_status()
    print(f"🔄 LangGraph Enabled: {status['langgraph_enabled']}")
    print(f"🤖 Available Agents: {', '.join(status['available_agents'])}")
    
except Exception as e:
    print(f"CRITICAL: Failed to initialize AlitaAgent: {e}")
    traceback.print_exc()
    agent = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/system_status', methods=['GET'])
def system_status():
    """Get system status including LangGraph capabilities."""
    if not agent:
        return jsonify({'error': "Alita Agent is not available."}), 503
    
    try:
        status = agent.get_workflow_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': f"Failed to get system status: {str(e)}"}), 500

@app.route('/explain_routing', methods=['POST'])
def explain_routing():
    """Explain workflow routing for a given query without executing it."""
    if not agent:
        return jsonify({'error': "Alita Agent is not available."}), 503

    data = request.json
    if not data or 'command' not in data:
        return jsonify({'error': 'No command provided.'}), 400
    
    command = data['command']
    if not command.strip():
        return jsonify({'error': 'Command cannot be empty.'}), 400

    try:
        explanation = agent.explain_workflow_routing(command)
        uses_langgraph = agent.langgraph_manager.should_use_langgraph(command)
        
        return jsonify({
            'explanation': explanation,
            'uses_langgraph': uses_langgraph,
            'workflow_type': 'Multi-Agent LangGraph' if uses_langgraph else 'Traditional Single-Agent'
        })
    except Exception as e:
        return jsonify({'error': f"Failed to explain routing: {str(e)}"}), 500

@app.route('/send_command', methods=['POST'])
def send_command():
    """Enhanced command processing with intelligent routing."""
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

    # Check if user wants to force a specific workflow type
    force_traditional = data.get('force_traditional', False)
    force_langgraph = data.get('force_langgraph', False)

    try:
        def generate_stream():
            try:
                if force_traditional:
                    # Force traditional single-agent processing
                    yield "🔧 **Forced Traditional Agent Processing**\n\n"
                    for chunk in agent.process_command_streaming(command):
                        if chunk != "quit_signal":
                            yield chunk
                elif force_langgraph:
                    # Force LangGraph multi-agent workflow
                    yield "🔄 **Forced Multi-Agent Workflow**\n\n"
                    try:
                        # Use async processing for LangGraph
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(agent._process_with_langgraph(command))
                        yield result
                        loop.close()
                    except Exception as e:
                        yield f"⚠️ Multi-agent workflow failed: {str(e)}\n"
                        yield "Falling back to traditional processing...\n\n"
                        for chunk in agent.process_command_streaming(command):
                            if chunk != "quit_signal":
                                yield chunk
                else:
                    # Use intelligent routing (default)
                    for chunk in agent.process_command_with_routing(command):
                        if chunk != "quit_signal":
                            yield chunk
            except Exception as e:
                yield f"Error during command processing: {str(e)}"
        
        return Response(stream_with_context(generate_stream()), mimetype='text/plain')
        
    except Exception as e:
        print(f"CRITICAL ERROR in /send_command for command '{command}': {e}")
        traceback.print_exc()
        return jsonify({'error': f"A critical server error occurred: {str(e)}"}), 500

@app.route('/send_command_async', methods=['POST'])
def send_command_async():
    """Async command processing endpoint."""
    if not agent:
        return jsonify({'error': "Alita Agent is not available."}), 503

    data = request.json
    if not data or 'command' not in data:
        return jsonify({'error': 'No command provided.'}), 400
    
    command = data['command']
    if not command.strip():
        return jsonify({'error': 'Command cannot be empty.'}), 400

    try:
        # Run async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(agent.process_command_async(command))
        loop.close()
        
        return jsonify({
            'response': result,
            'workflow_type': 'Multi-Agent' if agent.langgraph_manager.should_use_langgraph(command) else 'Traditional'
        })
        
    except Exception as e:
        print(f"ERROR in async command processing: {e}")
        traceback.print_exc()
        return jsonify({'error': f"Async processing failed: {str(e)}"}), 500

if __name__ == '__main__':
    if not agent:
        print("CRITICAL: AlitaAgent failed to initialize. Flask app cannot start.")
    else:
        print("🚀 Starting Enhanced Flask server with LangGraph support...")
        print("📍 Visit http://127.0.0.1:5001/ in your browser.")
        print("🔄 LangGraph multi-agent workflows are now available!")
        app.run(debug=True, port=5001)