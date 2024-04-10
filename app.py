import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

    return conversation_chain

def main():

    load_dotenv()
    ss = st.session_state

    st.set_page_config(page_title="Multi-PDF-Chat", page_icon=":books:")

    st.title("Multi-PDF-Chat :page_facing_up:")

    if "conversation_chain" not in ss:
        ss.conversation_chain = None
    if "chat_history" not in ss:
        ss.chat_history = None
    if "docs_processed" not in ss:
        ss.docs_processed = False

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type='pdf')
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)
                ss.docs_processed = True
                ss.conversation_chain = get_conversation_chain(vectorstore)

        if ss.docs_processed:
            st.text("Documents Processed")

    if prompt := st.chat_input("Ask questions about your documents here:"):
        
        # handle_user_input(prompt)
        response = ss.conversation_chain({'question': prompt})
        ss.chat_history = response['chat_history']
        for i, message in enumerate(ss.chat_history):
            if i % 2 == 0:
                with st.chat_message("user"):
                    st.markdown(message.content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(message.content)


if __name__ == '__main__':
    main()