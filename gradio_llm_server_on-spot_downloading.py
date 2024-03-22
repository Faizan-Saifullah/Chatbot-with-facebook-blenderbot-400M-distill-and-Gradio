import torch

from datasets import load_dataset
from transformers import GenerationConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def ask(text):
  
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to(device)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(**inputs, max_new_tokens=200, return_dict_in_generate=True)
    
    tokens = outputs.sequences.cpu().numpy()
    print(tokens)
    outputs = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    return outputs


## print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


import gradio as gr

with gr.Blocks() as server:
  with gr.Tab("LLM Inferencing"):

    model_input = gr.Textbox(label="Your Question:", value="", interactive=True)
    ask_button = gr.Button("Ask")
    model_output = gr.Textbox(label="The Answer:", interactive=False, value="")

  ask_button.click(ask, inputs=[model_input], outputs=[model_output])

server.launch()

