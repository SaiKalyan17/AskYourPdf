import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_perplexity import ChatPerplexity
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

def embedding_model():
    embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    return embedding

def llm_model():
    api_key = os.getenv('perplex_api_key')
    llm = ChatPerplexity(
        pplx_api_key = api_key,
        model="sonar",
        temperature=0.7
    )
    return llm

def get_pdf_data(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text
def get_doc_text(doc_files):
    text = ""
    for doc_file in doc_files:
        doc_reader = Document(doc_file)
        for para in doc_reader.paragraphs:
            text += para.text+"\n"
    return text
def get_data_chunks(raw_data):
    splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(raw_data)
    return chunks

def get_vector_data(data_chunks):
    vector_data = FAISS.from_texts(texts=data_chunks,embedding=embedding_model())
    return vector_data

def handle_conversation(vector_data):
    memory = ConversationBufferMemory(memory_key="chat_history",output_key="answer",return_messages=True)
    QA_prompt = PromptTemplate(
        input_variables=['context','question'],
        template="""You are an assistant for answering questions about the provided PDF documents.
Use only the following context to answer. 
If the answer is not contained in the context, say "I don't know."
Context:
{context}
Question: {question}
Answer:""")
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm = llm_model(),
        memory = memory,
        retriever = vector_data.as_retriever(),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_prompt},
        output_key="answer"
    )
    return conversational_chain
