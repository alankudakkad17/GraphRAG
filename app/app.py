import streamlit as st
import requests

API_URL = "http://api:8000"

st.title("GraphRAG Assistant 🧬")

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Document Knowledge Base")
    uploaded_file = st.file_uploader("Upload Medical PDF", type=["pdf"])
    
    if st.button("Process Document"):
        if uploaded_file:
            with st.spinner("Extracting Graph & Vector Entities..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_URL}/upload", files=files)
                if response.status_code == 200:
                    st.success("Knowledge Base Updated!")
                else:
                    st.error("Processing failed.")
        else:
            st.warning("Please upload a file first.")

# Main Chat Interface
st.header("Ask the Assistant")
question = st.text_input("Enter your medical query:")

if st.button("Submit Query"):
    if question:
        with st.spinner("Retrieving facts..."):
            response = requests.post(f"{API_URL}/ask", json={"question": question})
            if response.status_code == 200:
                answer = response.json().get("answer")
                st.write("**Answer:**")
                st.info(answer)
            else:
                st.error(response.json().get("answer", "An error occurred."))
