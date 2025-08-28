import os
from langchain_perplexity import ChatPerplexity
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SequentialChain, ConversationalRetrievalChain
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

load_dotenv()
api_key = os.getenv("perplex_api_key")
model = ChatPerplexity(
    model="sonar",
    pplx_api_key = api_key,
    temperature= 0.6
)

embedding_model = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-base"
)
# res = model.invoke("Hello How you are doing?")
# print(res)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
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
def get_data_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_stores(text_chunks):
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embedding_model)
    return vectorstore

def get_conversational_chain(vectorstore):

    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True, output_key="answer")
    QA_prompt = PromptTemplate(
        input_variables=['context','question'],
        template="""You are an assistant for answering questions about the provided PDF documents.
Use only the following context to answer. 
If the answer is not contained in the context, say "I don't know."
Context:
{context}
Question: {question}
Answer:"""
    )
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_prompt},
        output_key="answer"
    )
    return conversational_chain


# res = get_vectorStores(["Hello world"])
# print(res)

"""
language_prompt = PromptTemplate(
    input_variables=['language'],
    template="Suggest me 10 movies of {language} language. "
)
# res2 = language_prompt.format(language = "Telugu")
# print(res2)
rating_prompt = PromptTemplate(
    input_variables=['movies','rating','year'],
    template= "From the following movies: {movies}, "
        "which have IMDb rating greater than {rating} "
        "and were released after {year}? Return Movie names only as a list"
)
# print(rating_prompt.format(rating = "7",year = 2016))

chain1  = LLMChain(llm = model,prompt = language_prompt,output_key="movies")
chain2 = LLMChain(llm = model, prompt = rating_prompt,output_key="filtered_movies")

main_chain = SequentialChain(
    chains= [chain1,chain2],
    input_variables=["language", "rating", "year"],
    output_variables=["movies", "filtered_movies"],
    verbose= True
)
res = main_chain({
    "language": "Telugu",
    "rating": "7",
    "year": "2016"
})
print(res)

"""
