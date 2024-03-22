from transformers import AutoModelForCausalLM, AutoTokenizer

def initialize_model_and_tokenizer(model_name= "flan-t5-base"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model,tokenizer

model, tokenizer = initialize_model_and_tokenizer()


from langchain.llms.base import LLM

class CustomLLM(LLM):
    def _call(self, prompt, stop=None, run_manager=None) -> str:
        inputs = tokenizer(prompt, return_tensors="pt")
        result = model.generate(input_ids=inputs.input_ids, max_new_tokens=20)
        result = tokenizer.decode(result[0])
        return result

    @property
    def _llm_type(self) -> str:
        return "custom"

llm = CustomLLM()

from langchain import PromptTemplate

template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

from langchain import LLMChain

llm_chain = LLMChain(prompt=prompt, llm=llm)

import gradio as gr

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    llm_chain, llm = LLMChain.invoke(model, tokenizer)

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        print("Question: ", history[-1][0])
        bot_message = llm_chain.run(question=history[-1][0])
        # bot_message = "Hi"
        print("Response: ", bot_message)
        history[-1][1] = ""
        history[-1][1] += bot_message
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()

    