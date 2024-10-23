from flask import Flask, request, render_template, session # Ayy cuh install ts with pip first cuh
from models.llama3.api.datatypes import (
    UserMessage,
    CompletionMessage,
    SystemMessage,
    StopReason,
)
from models.llama3.reference_impl.generation import Llama
import os

app = Flask(__name__)
app.secret_key = 'your-secure-secret-key'  # Replace with a secure key in production

# Configuration parameters
ckpt_dir = r'C:\Users\prako\.llama\checkpoints\Llama3.1-8B'  # Ensure this is correct
temperature = 0.6
top_p = 0.9
max_seq_len = 512
max_batch_size = 4
max_gen_len = 150  # Adjusted for concise answers
model_parallel_size = None

# Initialize the Llama model
generator = Llama.build(
    ckpt_dir=ckpt_dir,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
    model_parallel_size=model_parallel_size,
)


@app.route('/', methods=['GET', 'POST'])
def chat():
    if 'dialog' not in session:
        # Initialize dialog with a system message
        system_prompt = "Anda adalah seorang tutor matematika tingkat SMA yang sangat baik, menjelaskan konsep matematika dengan mendalam menggunakan bahasa Indonesia dan gaya bicara Jaksel (Jakarta Selatan) yang santai dan kekinian."
        session['dialog'] = [{'role': 'system', 'content': system_prompt}]

    if request.method == 'POST':
        user_input = request.form['user_input'].strip()
        if user_input:
            # Append the user's message to the dialog
            session['dialog'].append({'role': 'user', 'content': user_input})

            # Prepare the dialog for the model
            dialog = []
            for msg in session['dialog']:
                if msg['role'] == 'user':
                    dialog.append(UserMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    dialog.append(CompletionMessage(
                        content=msg['content'],
                        stop_reason=StopReason.end_of_turn,
                    ))
                elif msg['role'] == 'system':
                    dialog.append(SystemMessage(content=msg['content']))

            # Generate the response
            result = generator.chat_completion(
                dialog,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            out_message = result.generation

            # Append the assistant's response to the dialog
            session['dialog'].append({'role': 'assistant', 'content': out_message.content})

        # Render the chat history
        return render_template('chat.html', dialog=session['dialog'])
    else:
        return render_template('chat.html', dialog=session.get('dialog', []))


@app.route('/reset', methods=['POST'])
def reset():
    session.pop('dialog', None)
    return render_template('chat.html', dialog=[])


if __name__ == '__main__':
    app.run(debug=True)
