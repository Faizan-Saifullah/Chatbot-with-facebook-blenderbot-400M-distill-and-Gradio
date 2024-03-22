from transformers.utils import logging
logging.set_verbosity_error()
from transformers import pipeline


""" 
First run these lines to download model and tokenizer to the device:

# chatbot = pipeline(task="conversational", model="facebook/blenderbot-400M-distill")
# model = chatbot.model
# tokenizer = chatbot.tokenizer
# model.save_pretrained("./saved_model")
# tokenizer.save_pretrained("./saved_tokenizer")

Then, comment the above lines and use the below lines: 
"""

def ask(model_input):

    chatbot = pipeline(task="conversational",model="./saved_model", tokenizer="./saved_tokenizer")
    # user_message = "What are some fun activities that I can do in winters?"
    user_message = model_input
    from transformers import Conversation
    conversation = Conversation(user_message)
    #un-comment to check what this prints
    # print(conversation)
    conversation = chatbot(conversation)
    print(conversation)

    ########to keep previous information/context in the chat so that the chatbot can respond in a human-like manner i.e. chatbot must remember the previous information.
    #-conversation.add_user_input("What else do you recommend?")
    ##### print(conversation)
    ###### conversation = Conversation(conversation)  >>> ## >>>  don't do this
    #-conversation = chatbot(conversation)
    #-print(conversation)

    return conversation


import gradio as gr

with gr.Blocks() as server:
  with gr.Tab("LLM Inferencing"):

    model_input = gr.Textbox(label="Your Question:", value="", interactive=True)
    ask_button = gr.Button("Ask")
    model_output = gr.Textbox(label="The Answer:", interactive=False, value="")

  ask_button.click(ask, inputs=[model_input], outputs=[model_output])

server.launch()

