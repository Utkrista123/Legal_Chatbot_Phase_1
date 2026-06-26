import sys
import time
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME =  "nepal_constitution"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
LLM_MODEL = "qwen3:8b"
TOP_K = 5

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(name = COLLECTION_NAME)

def check_ollama_ready(model_name):
    try:
        models_response = ollama.list()
        available_models = [m["model"] for m in models_response["models"]]

        model_found = any(
            model_name in m or m.startswith(model_name.split(":")[0])
            for m in available_models
        )

        if not model_found:
            return False, (
                f"Model '{model_name}' not found in Ollama.\n"
            )
        return True, ""
    
    except Exception as e:
        return False, (
            "Could not connect to Ollama. Is it running?\n"
            f"Technical error: {str(e)}"
        )
    
def retrieve_articles(query, embed_model, collection, top_k = TOP_K):
    query_embedding = embed_model.encode(
        f"query: {query}",
        normalize_embeddings = True
    )

    results = collection.query(
        query_embeddings = [query_embedding.tolist()],
        n_results = top_k,
        include = ["documents", "metadatas", "distances"]
    )

    articles = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        relevance = round((1 - distance / 2) * 100, 1)
        articles.append({
            "article_number": meta["article_number"],
            "text": doc,
            "relevance": relevance
        })
    return articles

def building_rag_prompt(query, articles):
    context_blocks = []
    for article in  articles:
        block = (
            f"[Article {article['article_number']}]\n"
            f"{article['text']}"
        )
        context_blocks.append(block)
    context_text = "\n\n---\n\n".join(context_blocks)

    system_prompt = """
                    You are a legal research assistant specializing in the Constitution of Nepal.

                    Rules you must follow:
                    1. Answer the question using ONLY the constitutional articles provided below.
                    2. Every sentence in your answer must cite  the article number it came from, like this: (Article 17).
                    3. If the provided articles do not contain enough information to answer the question, say exactly: "The provided
                    articles do not contain enough information to answer this question."
                    4. Do not use any outside knowledge about Nepal's constitution beyond what is given below.
                    5. Write in clear, plain English. Avoid unnecessary legal jargon.
                    6. Keep your answer concise -  3 to 6 sentences unless the question requires more detail.
                    """
    user_prompt = f"""CONTITUTIONAL ARTICLES PROVIDED AS CONTEXT:
                        {context_text}
                    QUESTION: {query}
                    Answer the question above using only the articles provided, with citations.
                    """
    return system_prompt, user_prompt
    
def generate_answer(system_prompt, user_prompt, model_name):
    response = ollama.chat(
        model = model_name,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={
            "temperature": 0.1,
        }
    )

    return response["message"]["content"]

# Streamlit page

st.set_page_config(
    page_title="Nepal Constitution Chatbot",
    layout = "wide"
)

st.title("Nepal Contitution Chatbot")
st.caption(f"Running **{LLM_MODEL}** locally")

# Load resources
with st.spinner("Loading embedding model and database..."):
    embed_model = load_embedding_model()
    collection = load_collection()

if collection is None:
    st.error("Database not found.")
    st.stop()

# check Ollama and the model are ready
ollama_ready, ollama_error = check_ollama_ready(LLM_MODEL)
if not ollama_ready:
    st.error(ollama_error)
    st.stop

st.success(f"Ready - {collection.count()} articles loaded.")
st.divider()

# Converstation history
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

# Display chat history
for msg in st.session_state.rag_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "articles" in msg:
            with st.expander(f"View {len(msg['articles'])} retrieved articles"):
                for a in msg["articles"]:
                    st.markdown(f"**Article {a['article_number']}** ({a["relevance"]}% relevant)")
                    st.markdown(a["text"])
                    st.divider()

# Chat input
user_query = st.chat_input("Ask a question about Nepal's constitution...")

if user_query:
    # 1. Show and save the user's question
    st.session_state.rag_messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Retrive relevant articles
    with st.chat_message("assistant"):
        with st.spinner(f"Searching constituiton articles..."):
            articles = retrieve_articles(user_query, embed_model, collection, TOP_K)

            if not articles:
                answer = "I couldn't find any relevant article for this question."
                st.markdown(answer)
                st.session_state.rag_message.append({"role": "assistnat", "content": answer})
            else:
                system_prompt, user_prompt = building_rag_prompt(user_query, articles)

                with st.spinner(f"Generating..."):
                    start_time = time.time()
                    answer = generate_answer(system_prompt, user_prompt, LLM_MODEL)
                    elapsed = round(time.time() - start_time, 1)
                
                st.markdown(answer)
                st.caption(f"Generated in {elapsed}s.")

        with st.expander(f"View {len(articles)} retrieved articles used for this answer."):
            for a in articles:
                st.markdown(f"**Article {a['article_number']}** ({a["relevance"]}% relevant)")
                st.markdown(a["text"])
                st.divider()

        st.session_state.rag_messages.append({
            "role": "assistant",
            "content": answer,
            "articles": articles
        })