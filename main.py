import streamlit as st
from model import get_pdf_text, get_data_chunks, get_vector_stores, get_conversational_chain


def main():
    st.set_page_config(page_title = "KnowYourPDF",page_icon=":books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with PDFs :books:")
    user_input = st.text_input("Ask a question about your document: ")
    if user_input:
        handle_user_input(user_input)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your pdfs here and click proceed ",accept_multiple_files=True)
        st.button("Proceed")
        with st.spinner("Processing..."):
            #Get Raw Data from PDF's
            raw_text = get_pdf_text(pdf_docs)
            # Split Raw Data into Chunks
            chunks = get_data_chunks(raw_text)
            # Store Embeddings of chunks into VectorDB
            vectorstore = get_vector_stores(chunks)
            #Start Conversational Chain
            st.session_state.conversation = get_conversational_chain(vectorstore)


def handle_user_input(user_text):
    res = st.session_state.conversation({'question':user_text})
    st.write(res["answer"])
main()