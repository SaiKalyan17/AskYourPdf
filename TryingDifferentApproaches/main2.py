
import streamlit as st

from model2 import get_pdf_data, get_data_chunks, get_vector_data, handle_conversation,get_doc_text

def main():
    st.set_page_config(page_title="KnowYourPDF", page_icon=":books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    st.header("Chat With PDFs Here:) :bike:")
    user_text = st.text_input("Ask a question about your document: ")
    if user_text:
        handle_input(user_text)

    with st.sidebar:
        st.header("Your documents")
        pdf_files = st.file_uploader("Upload your pdfs here and click proceed ", accept_multiple_files=True)
        st.button("Proceed")
        with st.spinner("Processing"):
            #Check if it's doc or pdf
            if pdf_files:
                for f in pdf_files:
                    # Get Text from PDFs
                    filetype = f.name.split('.')[1]
                    if filetype == "pdf":
                        raw_text = get_pdf_data(pdf_files)
                        data_chunks = get_data_chunks(raw_text)
                        # Add chunks embeddings into Vectordb
                        vectordata = get_vector_data(data_chunks)
                        # Create Retrieval System
                        st.session_state.conversation = handle_conversation(vectordata)
                    elif filetype == "docx":
                        raw_text = get_doc_text(pdf_files)
            # Make Raw data into Chunks



def handle_input(user_input):
    res = st.session_state.conversation({'question': user_input})
    st.write(res["answer"])

main()