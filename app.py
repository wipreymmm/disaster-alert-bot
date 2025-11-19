from flask import Flask, request, jsonify, render_template
from datetime import datetime
from model.rag_modelv4 import ask_question, refresh_web_data

# App
app = Flask(__name__)

# Routes
@app.route('/', endpoint='index') # Chatbot
def index():
    return render_template('index.html', title='DisasterAlertBot')

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    message = data.get("message", "").strip()
    
    if not message:
        return jsonify({"answer": "(Error: Empty message)"})
    
    try:
        answer = ask_question(message)  # call RAG function
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"(Error: {e})"})

@app.route("/refresh", methods=["POST"])
def refresh():
    try:
        refresh_web_data()
        return jsonify({"status": "success", "message": "Web sources refreshed!"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Refresh failed: {e}"})

if __name__ == "__main__":
    app.run(debug=True)

# Context Processors
@app.context_processor
def inject_now():
    contexts = {
        'now': datetime.now()
    }
    return contexts