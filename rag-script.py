from langchain_community.llms import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr
import shutil
import os

# RAG-test/rag_db
# /home/rocket/Desktop/projects/RAG-test
 
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectordb = Chroma(embedding_function=embeddings, persist_directory='/home/rocket/Desktop/projects/RAG-test/rag_db') #~/Desktop/projects/RAG-test/rag_db /home/ragadmin/rag/rag_db
llm = HuggingFaceEndpoint(
    huggingfacehub_api_token="YOUR KEY",
    endpoint_url="http://localhost:81", # removed v1
    max_new_tokens=1000, # shorted from 8000
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
 
 
def add_pdf_to_db(dir_name, dbname, chunk, overlap, model_name):
    location = "the document loading step"
    try:
        loader = PyPDFDirectoryLoader(dir_name)
        docs = loader.load()
        print("Documents loaded")
        location = "the text splitting step"
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk, chunk_overlap=overlap)
        all_splits = text_splitter.split_documents(docs)
        print("Documents split")
        location = "the embedding creation step"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        print("Embeddings created")
        location = "the vector database store step"
        vectordb = Chroma.from_documents(all_splits, embedding=embeddings, persist_directory=dbname)
        vectordb.persist()
        print("Documents added into the vector database")
        location = "the file cleanup step"
        for filename in os.listdir(dir_name):
            print(f"Filename {filename}")
            file_path = os.path.join(dir_name, filename)
            print(f"Checking {file_path}")
            if os.path.isfile(file_path) or os.path.islink(file_path):
                print(f"Unlinking {file_path}")
                os.unlink(file_path)
        print("Cleaned upload directory")
        return f"Document ingested into the vector database successfully"
    except Exception as exc:
        print(f"Error ingesting document at {location} because {exc}")
        return f"Error ingesting document at {location}"
 
def upload_file(file):
    shutil.copy(file, "/home/rocket/Desktop/projects/RAG-test/files")
    status = add_pdf_to_db("/home/rocket/Desktop/projects/RAG-test/files", "/home/rocket/Desktop/projects/RAG-test/rag_db", 500, 50, 'sentence-transformers/all-MiniLM-L6-v2')
    return status
 
def gen_response(message, history, rag_flag):
    vect_data = ""
    if rag_flag == False:
        resp = llm(message)
    else:
        resp = qa_chain({"query": message})['result'].strip()
        docs = vectordb.similarity_search(message)
        for doc in docs:
            vect_data += str(doc) + "\n\n"
    history.append((message, resp))
    return "", history, vect_data
 
def flag_change(rag_flag):
    if rag_flag == False:
        return gr.Textbox(visible=False)
    else:
        return gr.Textbox(visible=True)
 
 
with gr.Blocks(theme=gr.themes.Soft(), css="footer{display:none !important}") as demo:
    with gr.Row():
        with gr.Column(scale=0):
            gr.Image("/home/ragadmin/rag/dell_tech.png", scale=0, show_download_button=False, show_label=False, container=False) ### Possibly here
        with gr.Column(scale=4):
            gr.Markdown("")
    with gr.Row():
        chatbot = gr.Chatbot(scale=3)
        data = gr.Textbox(lines=17, max_lines=17, show_label=False, scale=1)
    prompt = gr.Textbox(container=False)
    with gr.Row():
        rag_flag = gr.Checkbox(label="Enable RAG")
        clear = gr.ClearButton([prompt, chatbot])
    with gr.Row():
        status = gr.Textbox("Ready for file upload", container=False)
        upload_button = gr.UploadButton("Click to Upload a PDF File", file_types=[".pdf"])
    upload_button.upload(upload_file, upload_button, status)
    prompt.submit(gen_response, [prompt, chatbot, rag_flag], [prompt, chatbot, data])
    rag_flag.change(flag_change, rag_flag, data)
demo.launch(server_name="0.0.0.0", server_port=7860)
