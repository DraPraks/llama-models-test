# Explanation of files in this dir
## app.py:
Webapp build on flask
## user_prompt.py
Script to ask for user input and show model output
## templates\index.html
html file to display for app.py

# HOW TO RUN:
1. Make sure you have the dependencies installed (refer to main README.md)
2. Make sure you've ran `pip install Flask Flask-SocketIO eventlet`
3. run app.py using `python app.py`

# TLDR:
## ii. Explanation
1. Flask and SocketIO: Flask serves the web pages, while SocketIO enables real-time bidirectional communication between the client and server.

1. Frontend: The HTML page contains a simple interface with a chat window, input field, and send button. JavaScript handles sending and receiving messages via SocketIO.

1. Backend: The server listens for incoming messages, processes them using the Llama model, and sends back the responses.

`Open a web browser and navigate to http://127.0.0.1:5000/ to start chatting with the model.`