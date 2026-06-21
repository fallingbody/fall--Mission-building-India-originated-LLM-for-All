"""
FALL Chat - Web-based chat interface.
Simple Flask app with HTML/CSS/JS frontend.
"""
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import requests
import json
import os
import uuid
import time

app = Flask(__name__)
CORS(app)

# Configuration
FALL_API_URL = os.environ.get("FALL_API_URL", "http://localhost:8000")
FALL_API_KEY = os.environ.get("FALL_API_KEY", "sk-local-dev")

class ChatSession:
    def __init__(self):
        self.sessions = {}
    
    def create(self):
        session_id = str(uuid.uuid4())[:12]
        self.sessions[session_id] = {
            "messages": [],
            "created_at": time.time(),
        }
        return session_id
    
    def add_message(self, session_id, role, content):
        if session_id in self.sessions:
            self.sessions[session_id]["messages"].append({
                "role": role,
                "content": content,
            })
    
    def get_history(self, session_id):
        if session_id in self.sessions:
            return self.sessions[session_id]["messages"]
        return []

sessions = ChatSession()


@app.route('/')
def index():
    """Serve the chat interface."""
    return render_template('chat.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id')
    mode = data.get('mode', 'interactive')
    
    if not session_id:
        session_id = sessions.create()
    
    # Add user message
    sessions.add_message(session_id, "user", message)
    
    try:
        # Call FALL API
        payload = {
            "task": message,
            "mode": mode,
            "tools": ["shell", "python", "file_read", "web_search"] if mode == "autonomous" else None,
            "max_duration": 300,
        }
        
        response = requests.post(
            f"{FALL_API_URL}/v1/execute",
            headers={
                "X-API-Key": FALL_API_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
        )
        
        if response.status_code == 200:
            result = response.json()
            assistant_message = result.get("result", {}).get("output", "")
            
            # Fallback: extract text from result
            if not assistant_message:
                assistant_message = str(result.get("result", "Task completed."))
        else:
            assistant_message = f"Error: API returned {response.status_code}"
        
    except requests.exceptions.ConnectionError:
        assistant_message = "Error: Cannot connect to FALL API. Is the server running?"
    except requests.exceptions.Timeout:
        assistant_message = "Error: Request timed out."
    except Exception as e:
        assistant_message = f"Error: {str(e)}"
    
    # Add assistant message
    sessions.add_message(session_id, "assistant", assistant_message)
    
    return jsonify({
        "session_id": session_id,
        "message": assistant_message,
        "status": "ok",
    })


@app.route('/api/stream', methods=['POST'])
def stream_chat():
    """Stream chat responses."""
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id')
    
    if not session_id:
        session_id = sessions.create()
    
    sessions.add_message(session_id, "user", message)
    
    def generate():
        try:
            payload = {
                "task": message,
                "mode": "interactive",
            }
            
            response = requests.post(
                f"{FALL_API_URL}/v1/execute/stream",
                headers={"X-API-Key": FALL_API_KEY, "Content-Type": "application/json"},
                json=payload,
                stream=True,
                timeout=120,
            )
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            token = chunk.get("token", "")
                            full_response += token
                            yield f"data: {json.dumps({'token': token})}\n\n"
                        except json.JSONDecodeError:
                            pass
            
            sessions.add_message(session_id, "assistant", full_response)
            yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        }
    )


@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get chat history."""
    history = sessions.get_history(session_id)
    return jsonify({
        "session_id": session_id,
        "messages": history,
    })


@app.route('/api/sessions/new', methods=['POST'])
def new_session():
    """Create a new session."""
    session_id = sessions.create()
    return jsonify({"session_id": session_id})


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  FALL Chat Interface")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)