import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import json

def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def create_knowledge_base(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embeddings)

def get_answer_from_chain(knowledge_base, user_question):
    docs = knowledge_base.similarity_search(user_question)
    llm = OpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=docs, question=user_question)

def format_output_as_json(questions, answers):
    output = {question: answers.get(question, "Data Not Available") for question in questions}
    return json.dumps(output, indent=2)

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF", layout="centered", initial_sidebar_state="auto")
    st.header("PDF Question Answer (OpenAI) 🤓")
    
    # Uploading the file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # Extracting the text
    if pdf is not None:
        text = extract_text_from_pdf(pdf)
        chunks = chunk_text(text)
        knowledge_base = create_knowledge_base(chunks)
        
        # Show user input
        questions = st.text_area("Enter your questions (one per line):").splitlines()
        
        if st.button("Get Answers"):
            if questions:
                answers = {}
                for question in questions:
                    response = get_answer_from_chain(knowledge_base, question)
                    # Here you might want to check confidence and format response accordingly
                    answers[question] = response if response else "Data Not Available"
                
                # Display the results
                output = format_output_as_json(questions, answers)
                st.json(output)
            else:
                st.warning("Please enter at least one question.")
                
        if st.button("Refresh Page"):
            st.experimental_rerun()

if __name__ == '__main__':
    main()
