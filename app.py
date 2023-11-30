import arxiv
import gradio
from llama_index import (
        VectorStoreIndex,
        ServiceContext,
        SimpleDirectoryReader,
        Document
)
from langchain.llms import HuggingFaceHub
from llama_index.llms import LangChainLLM

repo_id = 'HuggingFaceH4/zephyr-7b-beta'

def loadind_paper()
    return 'Loading...'

def paper_changes(paper_id):
    paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id]))))
    docs = SimpleDirectoryReader(input_files=[paper.download_pdf()]).load_data()
    doc = Document(text='\n\n'.join([doc.text for doc in docs]))
    llm = LangChainLLM(llm=HuggingFaceHub(
        repo_id=repo_id, 
        model_kwargs={
            'temperature': 0.2,
            'max_tokens': 4096,
            'top_p': 0.9
        }
    )
    service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_documents([doc], service_context=service_context)
    global query_engine
    query_engine = index.as_query_engine()
    return 'Ready!!!'

def add_text(history, text):
    history = history + [(text, None)]
    return history, ''

def bot(history):
    response = infer(history[-1][0])
    history[-1][1] = response
    return history

def infer(question):
    response = query_engine.query(question)
    return str(response)

with gr.Blocks(theme='WeixuanYuan/Soft_dark') as demo:
    with gr.Column():
        chatbot = gr.Chatbot([], elem_id='chatbot')

        with gr.Row():
            paper_id = gr.Textbox(label='ArXiv Paper Id', placeholder='1706.03762')
            langchain_status = gr.Textbox(label='Status', placeholder='', interactive=False)
            load_paper = gr.Button('Load Paper to LLaMa-Index')

        with gr.Row():
            question = gr.Textbox(label='Question', placeholder='Type your query...')
            submit_btn = gr.Button('Submit')

    load_paper.click(paper_changes, inputs=[paper_id], outputs=[langchain_status], queue=False)
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(bot, chatbot, chatbot)
    submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(bot, chatbot, chatbot)

demo.launch(share=True)
