from transformers import AutoModelForCausalLM, AutoTokenizer

def initialize_model_and_tokenizer(model_name="google/flan-t5-base"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = initialize_model_and_tokenizer()

from langchain.llms.base import LLM

class CustomLLM(LLM):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
    
    def _call(self, prompt, stop=None, max_tokens=20, run_manager=None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        result = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(result[0], skip_special_tokens=True)

    @property
    def _llm_type(self) -> str:
        return "custom"

# Initialize the custom LLM with the model and tokenizer
llm = CustomLLM(model, tokenizer)

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

template = "Question: {question}\nAnswer: Let's think step by step."
prompt = PromptTemplate(template=template, input_variables=["question"])

# Initialize the LLMChain with the custom LLM and prompt template
llm_chain = LLMChain(prompts=[prompt], llm=llm)

import gradio as gr

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        if history:
            print("Question: ", history[-1][0])
            bot_message = llm_chain.run(question=history[-1][0])
            print("Response: ", bot_message)
            history[-1][1] = bot_message
        return history

    msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(bot, inputs=chatbot, outputs=chatbot)
    clear.click(lambda: chatbot.reset(), inputs=None, outputs=chatbot, queue=False)

demo.launch()
