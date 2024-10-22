import streamlit as st
import pandas as pd
import openai
import os
import tempfile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Set the Streamlit page configuration
st.set_page_config(layout="wide", page_icon="💬", page_title="ChatBot-CSV")

# Display the header for the application using HTML markdown
st.markdown(
    "<h1 style='text-align: center;'>ChatBot-CSV, Talk with your csv-data ! 💬</h1>",
    unsafe_allow_html=True)

# Allow the user to enter their OpenAI API key
user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

def load_csv(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def store_doc_embeds(file, filename):
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
        tmp_file.write(file)
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
    data = loader.load()
    embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(data, embeddings)
    os.remove(tmp_file_path)

    # Save the vectors to a temporary file instead of pickling to avoid thread issues
    vectors.save_local(filename + "_index")

def get_doc_embeds(file, filename):
    if not os.path.isfile(filename):
        store_doc_embeds(file, filename)

    # Load the vectors from the saved file
    vectors = FAISS.load_local(filename + '_index', OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    return vectors

def conversational_chat(query, chain):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def main():
    if user_api_key == "":
        st.markdown(
            "<div style='text-align: center;'><h4>Enter your OpenAI API key to start chatting 😉</h4></div>",
            unsafe_allow_html=True)
    else:
        os.environ["OPENAI_API_KEY"] = user_api_key
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv", label_visibility="visible")

        if uploaded_file is not None:
            def show_user_file(uploaded_file):
                file_container = st.expander("Your CSV file :")
                shows = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)
                file_container.write(shows)
            show_user_file(uploaded_file)

            try:
                with st.spinner("Processing..."):
                    uploaded_file.seek(0)
                    file = uploaded_file.read()
                    vectors = get_doc_embeds(file, uploaded_file.name.replace(' ', '_').replace('(', '').replace(')', ''))
                    chain = ConversationalRetrievalChain.from_llm(
                        llm=ChatOpenAI(temperature=0.0, model_name='gpt-4', streaming=True),
                        retriever=vectors.as_retriever(), chain_type="stuff"
                    )
                st.session_state['ready'] = True

                if st.session_state['ready']:
                    if 'history' not in st.session_state:
                        st.session_state['history'] = []

                    response_container = st.container()
                    container = st.container()

                    with container:
                        with st.form(key='my_form', clear_on_submit=True):
                            user_input = st.text_input("Query:", placeholder="Talk about your CSV data here (:", key='input')
                            submit_button = st.form_submit_button(label='Send')

                            if submit_button and user_input:
                                output = conversational_chat(user_input, chain)
                                st.write(f"**User**: {user_input}")
                                st.write(f"**Bot**: {output}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

        else:
            st.sidebar.info(
                "👆 Upload your CSV file to get started, "
            )

    about = st.sidebar.expander("About 🤖")
    about.write("#### ChatBot-CSV is an AI chatbot featuring conversational memory, designed to enable users to discuss their CSV data in a more intuitive manner. 📄")
    about.write("#### It employs large language models to provide users with seamless, context-aware natural language interactions for a better understanding of their CSV data. 🌐")
    about.write("#### Powered by [Langchain](https://github.com/hwchase17/langchain), [OpenAI](https://platform.openai.com/docs/models/gpt-3-5) and [Streamlit](https://github.com/streamlit/streamlit) ⚡")

if __name__ == "__main__":
    main()