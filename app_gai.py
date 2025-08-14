import streamlit as st
import os
import time
import pickle

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Initialisation
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
DB_FAISS_PATH = "vectorstore/faiss_index"

# UI
st.set_page_config(page_title="📄 Q&A avec Groq et FAISS", layout="wide")
st.title("📄 Chatbot Intelligent sur Documents PDF")
st.markdown("""
    **Pose une question sur les documents PDF que tu uploades.**  
    Le chatbot te répondra en utilisant les documents que tu as chargés.  
    📚 Commence par uploader des fichiers PDF, puis pose ta question !
""")

# LLM
llm = ChatGroq(model_name="llama3-8b-8192", api_key=groq_api_key)

prompt = ChatPromptTemplate.from_template("""
Réponds à la question en te basant uniquement sur le contexte fourni ci-dessous.
<context>
{context}
</context>
Question : {input}
""")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Upload des PDF
uploaded_files = st.file_uploader("📤 Upload un ou plusieurs fichiers PDF", type=["pdf"], accept_multiple_files=True)

# Afficher les noms des fichiers uploadés
if uploaded_files:
    st.markdown("### Fichiers uploadés :")
    for file in uploaded_files:
        st.markdown(f"🗂️ **{file.name}**")

# Fonction : créer ou recharger l’index FAISS
def load_or_create_faiss(files):
    os.makedirs("vectorstore", exist_ok=True)

    if os.path.exists(DB_FAISS_PATH):
        with open(f"{DB_FAISS_PATH}.pkl", "rb") as f:
            vectorstore = pickle.load(f)
        st.session_state.vectors = vectorstore
        return "Index FAISS rechargé depuis le disque."
    else:
        all_docs = []
        for file in files:
            with open(os.path.join("vectorstore", file.name), "wb") as f:
                f.write(file.getbuffer())
            loader = PyPDFLoader(os.path.join("vectorstore", file.name))
            docs = loader.load()
            all_docs.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(all_docs)
        vectorstore = FAISS.from_documents(final_documents, embeddings)

        with open(f"{DB_FAISS_PATH}.pkl", "wb") as f:
            pickle.dump(vectorstore, f)

        st.session_state.vectors = vectorstore
        return "Index FAISS créé et sauvegardé."

# Embedding des documents
if uploaded_files and st.button("📚 Indexer les documents"):
    with st.spinner("Indexation des documents en cours..."):
        msg = load_or_create_faiss(uploaded_files)
        st.success(msg)

# Historique de conversation
if "history" not in st.session_state:
    st.session_state.history = []

# Question de l'utilisateur
question = st.text_input("💬 Pose ta question ici")

# Répondre à la question
if question and "vectors" in st.session_state:
    with st.spinner("Recherche et génération de réponse..."):
        try:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.time()
            response = retrieval_chain.invoke({"input": question})
            elapsed_time = round(time.time() - start, 2)

            # Affichage
            st.markdown(f"✅ **Réponse :** {response['answer']}")
            st.markdown(f"⏱️ Temps de réponse : `{elapsed_time} sec`")

            # Historique
            st.session_state.history.append((question, response['answer']))

            # Documents similaires
            with st.expander("📄 Voir les documents similaires utilisés"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Chunk {i+1} :**")
                    st.markdown(doc.page_content)
                    st.write("---")

        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
elif question:
    st.warning("⚠️ Veuillez d'abord indexer les documents.")

# Affichage de l'historique
if st.session_state.history:
    with st.expander("🕘 Historique des questions / réponses"):
        for i, (q, a) in enumerate(st.session_state.history[::-1]):
            st.markdown(f"**Q{i+1} :** {q}")
            st.markdown(f"**A{i+1} :** {a}")
            st.write("---")
