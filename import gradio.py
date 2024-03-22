# import gradio as gr
# import random
# import time

# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot()
#     msg = gr.Textbox()
#     clear = gr.Button("Clear")

#     def user(user_message, history):
#         return "", history + [[user_message, None]]

#     def bot(history):
#         bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
#         history[-1][1] = ""
#         for character in bot_message:
#             history[-1][1] += character
#             time.sleep(0.05)
#             yield history

#     msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
#         bot, chatbot, chatbot
#     )
#     clear.click(lambda: None, None, chatbot, queue=False)
    
# demo.queue()
# demo.launch()


"""
------ WE CAN ALSO LIKE DISLIKE MESSAGES IN THE CHAT:
"""

import gradio as gr

def greet(history, input):
    return history + [(input, "Hello, " + input)]

def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value)
    else:
        print("You downvoted this response: " + data.value)
    

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    textbox = gr.Textbox()
    textbox.submit(greet, [chatbot, textbox], [chatbot])
    chatbot.like(vote, None, None)  # Adding this line causes the like/dislike icons to appear in your chatbot
    
demo.launch()