from transformers.utils import logging
logging.set_verbosity_error()
from transformers import pipeline, Conversation


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
import time
# with gr.Blocks() as server:
#   with gr.Tab("LLM Inferencing"):

#     model_input = gr.Textbox(label="Your Question:", value="", interactive=True)
#     ask_button = gr.Button("Ask")
#     model_output = gr.Textbox(label="The Answer:", interactive=False, value="")

#   ask_button.click(ask, inputs=[model_input], outputs=[model_output])

# server.launch()
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
      chatbot = pipeline(task="conversational", model="./saved_model", tokenizer="./saved_tokenizer")
      conversation = Conversation(message)
      conversation = chatbot(conversation)
      print(conversation)

      # Extracting the last bot message
      # Assuming the conversation has at least one exchange
      if conversation.generated_responses:
         bot_message = conversation.generated_responses[-1]  # Get the last response from the bot
      else:
         bot_message = "Sorry, I didn't get that."  # Fallback message

      # Appending a tuple of (user_message, bot_message) to chat_history
      chat_history.append((message, bot_message))
    
      time.sleep(2)  # Simulating delay for bot response
    
      # Return an empty string for the first return value as expected by your GUI logic
      return "", bot_message


    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()
