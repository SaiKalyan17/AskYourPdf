# 📚 KnowYourDocs – Chat With Your Documents  

An AI-powered Streamlit app that lets you **chat with your PDF and DOCX files**.  
Upload a document, ask questions in natural language, and get precise answers extracted from your file.  

---

## ✨ Features
- 📂 Upload **PDF** and **Word (.docx)** files.  
- 🔍 Text extraction from documents.  
- 🧩 Smart text chunking for efficient retrieval.  
- 🧠 Embedding + Vector DB (FAISS) for fast similarity search.  
- 🤖 Conversational AI (LangChain + LLM) with memory.  
- 🎯 Context-aware answers: if info is missing, the AI will say *"I don't know."*

---

## ⚙️ Tech Stack
- [Streamlit](https://streamlit.io/) – UI framework  
- [LangChain](https://www.langchain.com/) – LLM & retrieval pipeline  
- [FAISS](https://github.com/facebookresearch/faiss) – Vector search engine  
- [PyPDF2](https://pypi.org/project/pypdf2/) – Extract text from PDFs  
- [python-docx](https://pypi.org/project/python-docx/) – Extract text from Word documents  
- [HuggingFace Instruct Embeddings](https://huggingface.co/hkunlp/instructor-base) – Embedding model  
- [ChatPerplexity API](https://www.perplexity.ai/) – LLM backend  

---

## 📦 Installation

Clone the repository:
```bash
git clone https://github.com/your-username/knowyourdocs.git
cd knowyourdocs
