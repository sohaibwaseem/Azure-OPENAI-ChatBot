import streamlit as st
from ChatBot import final_result
from streamlit_chat import message
import textract
import os
import glob
import pandas as pd
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain


os.environ['OPENAI_API_KEY'] = 'a194af7e52e64d22b86f65ac97bc0fd9'
os.environ['OPENAI_API_TYPE'] = 'azure'
os.environ['OPENAI_API_VERSION'] = '2023-09-15-preview'
os.environ['OPENAI_API_BASE'] = "https://vipindemo.openai.azure.com/"

embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002', 
                                deployment='vipin-ada',
                                openai_api_key = os.environ['OPENAI_API_KEY'],
                                openai_api_type=  os.environ['OPENAI_API_TYPE'],
                                openai_api_version= os.environ['OPENAI_API_VERSION'],
                                openai_api_base= os.environ['OPENAI_API_BASE'])

# def embeddings(pdf_path):
#     db = get_embeddings(pdf_path)
#     return db
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]


def get_bot_response(user_input, db):
    response = final_result(user_input, db)
    return response
    
# st.title("Simple Chatbot")
# pdf_path = st.text_input("Enter the path of your pdf file: ")

# uploaded_files = st.file_uploader("Upload multiple files", accept_multiple_files=True)

st.title("Chat with your own PDF files Using Azure OPENAI")
st.sidebar.title("Select the topic for Chating")
with open ("csv.txt","r") as f:
    names=f.readlines()
names=set(names)
print(names,"ggooo")
# if selected_option:


selected_option = st.sidebar.selectbox('Select the topic on which you want to Chat:', names)
st.sidebar.write('You selected topic is:', selected_option)
print("husfkntiwkrf", selected_option)
a = FAISS.load_local(selected_option.strip(), embeddings)

#
# files = []
# uploaded_files = st.sidebar.file_uploader("Uploaded files", accept_multiple_files=True, type="pdf")
# print(uploaded_files)
# if len(uploaded_files) != 0:
#     chunks_st = []
#     for names in uploaded_files:
#         files.append(names.name)
#         print("=============================")
#     for file in files:
#         doc = textract.process('/home/abdullah/Downloads/' + file)

#         with open('attention_is_all_you_need.txt', 'w') as f:
#             f.write(doc.decode('utf-8'))

#         with open('attention_is_all_you_need.txt', 'r') as f:
#             text = f.read()

#         tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
#         def count_tokens(text: str) -> int:
#             return len(tokenizer.encode(text))

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size = 512,
#             chunk_overlap  = 100,
#             length_function = count_tokens,
#         )

#         chunks = text_splitter.create_documents([text])
#         for i in chunks:
#             chunks_st.append(i.page_content)

#     embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002', 
#                                   deployment='vipin-ada',
#                                openai_api_key = os.environ['OPENAI_API_KEY'],
#                                openai_api_type=  os.environ['OPENAI_API_TYPE'],
#                                 openai_api_version= os.environ['OPENAI_API_VERSION'],
#                                 openai_api_base= os.environ['OPENAI_API_BASE'])

#     # Create vector database
#     db = FAISS.from_texts(chunks_st[:1], embeddings)
#     for i in range(1,len(chunks_st),16):
#         db.add_texts(chunks_st[i:i+16])

initialize_session_state()
# def display_chat_history(chain):
reply_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Enter your message here:")
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        with st.spinner('Generating response...'):
            output = get_bot_response(user_input, a)

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with reply_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
            message(st.session_state["generated"][i], key=str(i), avatar_style="Minimalist")        

        
# user_input = st.text_input("Enter your message here:") 

# if st.button("Send"):
#     if user_input:
#         bot_response = get_bot_response(user_input, db)
#         st.success(bot_response)
#     else:
#         st.warning("Please enter a message")
