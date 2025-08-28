
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Configura tu clave de OpenAI
openai_api_key = st.secrets["OPENAI_API_KEY"]

st.title("Agente de consulta de PDFs")

uploaded_file = st.file_uploader("Sube un documento PDF", type="pdf")
query = st.text_input("Haz una pregunta sobre el documento")

if uploaded_file and query:
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_texts(texts, embeddings)

    docs = db.similarity_search(query)
    llm = OpenAI(openai_api_key=openai_api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)

    st.write("Respuesta:")
    st.write(answer)
