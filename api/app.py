# app.py (Modified)
from flask import Flask,render_template,request,jsonify
from chatbot import get_response # Imports your function from chatbot.py

# Initialize the Flask web application
app = Flask(__name__)


@app.route("/")
def home():
    # This serves the main chat page to the browser
    return render_template("index.html")

@app.route("/get_response", methods=['POST'])
def get_bot_response():
    user_text = request.form['msg'] # Get data sent from the form
    return get_response(user_text)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)