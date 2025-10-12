import streamlit as st
import requests

    
BACKEND_URL = "http://localhost:8000/query"

st.set_page_config(page_title="Knowledge-Base Search Engine", layout="centered")
st.title("📚 Knowledge-Base Search Engine")
st.write("Upload documents and ask questions. Answers are generated using RAG + LLM.")

# User query input
query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                response = requests.post(BACKEND_URL, json={"query": query})
                if response.status_code == 200:
                    data = response.json()
                    st.subheader("Answer:")
                    st.write(data["answer"])
                else:
                    st.error(f"Error: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
