# pdf-question-answering-openai
Description: A Streamlit application for extracting answers from PDF documents using OpenAI's GPT-3.5-turbo. This project leverages LangChain for text chunking and FAISS for efficient similarity search.



# PDF Question Answering with OpenAI

## Overview
A Streamlit application that allows users to upload a PDF, ask questions, and receive answers based on the content of the PDF using OpenAI's GPT-3.5-turbo. The application uses LangChain for text processing and FAISS for efficient similarity search.

## Features
- Upload PDF documents and extract text.
- Ask questions related to the content of the PDF.
- Receive context-based answers using OpenAI's GPT-3.5-turbo.
- Answers are derived directly from the PDF content.
- Handles low-confidence answers by returning "Data Not Available."

## Requirements
- Python 3.x
- Streamlit
- LangChain
- PyPDF2
- OpenAI
- python-dotenv

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdf-question-answering-openai.git

