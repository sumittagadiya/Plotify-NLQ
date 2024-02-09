__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import google.generativeai as genai
import pdfplumber
import os
from dotenv import load_dotenv
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from chromadb.config import Settings
import chromadb
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Initialized Logger
logging.basicConfig(
    filename= "pdf_qa.log",
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: \
        %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %Z",
)

# this method is not using text chunking and splitting, it is giving whole pdf as a context
def get_gemini_response(pdf_text, question,generative_model):
    #if uploaded_file is not None:
    try:
        # with pdfplumber.open(uploaded_file) as pdf:
        #     pages = [page.extract_text() for page in pdf.pages]
        #     pdf_text = "\n".join(pages)
        logging.info('Called get_gemini_response() method')
        input_prompt = f"""
            You are an AI trained to answer questions about PDF documents.

            Here is a PDF document: {pdf_text}

            Question: {question}
            Answer : 
            Reasoning:
            
            Answer the question accurately based on the information in the PDF document. Provide a clear and concise response along with Answer. Do not include Question in return response.
            """
        
        response = generative_model.generate_content(
            [input_prompt]
            # max_tokens=200, temperature=0.7
        )
        logging.info(f"\n Response : {response.text}")
        return response.text
        #status.update(label="Response", state="complete")
    except Exception as e:
        logging.exception(str(e))
        raise Exception(str(e))


# This method will use Retrieval Augmented Generation method to get the relevant pasage from the pdf and pass it as a context to LLM for QA
def get_RAG_gemini_response(pdf_text,question):
    try:
        logging.info('Called RAG_get_gemini_response() method')
        #if uploaded_file is not None:
        #documents = PyPDFLoader(pdf_file).load()
        # with pdfplumber.open(uploaded_file) as pdf:
        #     pages = [page.extract_text() for page in pdf.pages]
        # documents = "\n".join(pages)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
        texts = text_splitter.split_text(pdf_text)
        logging.info(f"length of splitted text = {len(texts)}")

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        chroma_client = chromadb.Client(Settings(anonymized_telemetry=False,allow_reset=True))
        chroma_client.reset()
        db = Chroma.from_texts(texts, embeddings,client=chroma_client)
        vector_index = db.as_retriever(search_type="similarity", search_kwargs={"k":4},return_source_documents=False)

        docs = vector_index.get_relevant_documents(question)
        logging.info(f"Length of retrieved docs = {len(docs)}")
        
        prompt_template = """You are a helpful and informative bot that answers questions using text from the reference context included below. \
                            Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
                            However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
                            strike a friendly and converstional tone. \
                            If the passage is irrelevant to the answer, you may ignore it.
                            QUESTION: '{question}'
                            CONTEXT: '{context}'

                            ANSWER:
                            """

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        model = ChatGoogleGenerativeAI(model="gemini-pro")
        
        stuff_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        stuff_answer = stuff_chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        logging.info(f"\n Response : {stuff_answer['output_text']}")
        #response = generative_model.generate_content([prompt_template])
        #return response.text
        return stuff_answer['output_text']
    
    except Exception as e:
        logging.exception(str(e))
        raise Exception(str(e))

# Create an instance of the Gemini Pro model
# generative_model = genai.GenerativeModel('gemini-pro')

st.set_page_config(page_title='PDF Q&A App')
# App layout
st.title("Document Q&A with Google Gemini Pro")
logging.info("\n*********************** LOGGING STARTED **********************************\n")
uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])

# Query text
user_question = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

#with st.form('myform', clear_on_submit=True):
submitted = st.button('Get an Answer', disabled=not(uploaded_file and user_question))

if submitted:
    logging.info(f"File Name : {uploaded_file.name}")
    logging.info(f"Question : {user_question}")

    with st.spinner('Fetching an answer...'):
        try:
            if uploaded_file.type == "application/pdf":
                with pdfplumber.open(uploaded_file) as pdf:
                    pages = [page.extract_text() for page in pdf.pages]
                documents = "\n".join(pages)

            elif uploaded_file.type == "text/plain":
                # For a text file, simply read its contents
                documents = str(uploaded_file.read(), 'utf-8')
                
            generative_model = genai.GenerativeModel('gemini-pro')
            token_count = generative_model.count_tokens(documents).total_tokens

            logging.info(f'Total tokens in pdf = {token_count}')

            if token_count < 32000:
                answer = get_gemini_response(documents,user_question,generative_model)
            else:
                answer = get_RAG_gemini_response(documents,user_question)
        
            st.write("**Response:**")
            st.markdown(f"<p style='white-space: normal;'>{str(answer)}</p>", unsafe_allow_html=True)

        except Exception as e:
            logging.exception(str(e))
            st.error(f"Error: {e}")

