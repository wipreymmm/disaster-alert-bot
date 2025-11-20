from flask import Flask, request, jsonify, render_template
from model.rag_modelv4 import ask_question, refresh_web_data
from datetime import datetime
import logging

# App
app = Flask(__name__)

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Context Processors
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

# Routes
@app.route('/', endpoint='index')
def index():
    return render_template('index.html', title='DisasterAlertBot')

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        
        if not data:
            logger.error("No JSON data received")
            return jsonify({"answer": "Error: No data received"}), 400
        
        message = data.get("message", "").strip()
        
        if not message:
            logger.warning("Empty message received")
            return jsonify({"answer": "Please enter a question."}), 400
        
        logger.info(f"Received question: {message}")
        
        answer = ask_question(message)
        
        logger.info(f"Generated answer: {answer[:100]}...")
        
        return jsonify({"answer": answer}), 200
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        return jsonify({"answer": "Sorry, I encountered an error. Please try again."}), 500

@app.route("/refresh", methods=["POST"])
def refresh():
    try:
        logger.info("Refreshing web data...")
        refresh_web_data()
        logger.info("Web data refreshed successfully")
        return jsonify({"status": "success", "message": "Web sources refreshed!"}), 200
    except Exception as e:
        logger.error(f"Error refreshing data: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": f"Refresh failed: {str(e)}"}), 500

# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Run app
if __name__ == "__main__":
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000
    )