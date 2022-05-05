"""
This API is created for connecting to GPT-3 and to miro.com.
GPT-3 is API for OpenAI 
miro.com is a website that is used for team collaboration.
"""

import os
import json
from typing import Union
from flask import (
    Flask,
    request,
)
import gpt3_chat

app = Flask(__name__)

@app.route("/gpt3", methods=["POST"])
def house_api() -> dict:
    """
    This API connects to GPT-3 and asks a question
    :return: Dictionary of the answer from GPT-3
    """
    content_json = request.get_json()
    testing_mode = False
    answer = ""
    if "test" in content_json:
        testing_mode = bool(content_json["test"])
        print("testing mode activated")

    if "query" in content_json:
        query = content_json["query"]
        if testing_mode:
            answer = "Heck, this is working! (test mode activated)"
        else:
            answer = gpt3_chat.gpt_3_chat(query)
        return {"answer": answer}
    return {"error": "please, follow the instructions {'query': 'your query for gpt-3'}"}


@app.route("/", methods=["GET"])
def api_alive() -> dict:
    """
    Check if the API is alive
    :return: Dictionary with the status of the API
    """
    return {"status": "alive"}

@app.route("/gpt3", methods=["GET"])
def api_instructions() -> dict:
    """
    Returns instructions how to call the API
    :return: Dictionary with the status of the API
    """
    return {"question": "string: what ever you want to ask from OpenAI GPT-3", 
            "test": "boolean: true you want only to test the API"}

if __name__ == "__main__":
    app.run(debug=True)
    port = os.environ.get("PORT", 5001)
    app.run(host="0.0.0.0", port=port)
