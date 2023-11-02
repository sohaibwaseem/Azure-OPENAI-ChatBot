import streamlit as st
from ChatBot import final_result
from streamlit_chat import message
import textract
import subprocess
import os
import glob
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import  Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
# import pinecone
import os
import openai
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


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

def create_embeddings():
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    st.markdown("## New Project")
    st.title("Enter Project Details")
    project_name = st.text_input("Enter Project Name")
    print("project_name",project_name)
    if project_name:
        with open("csv.txt","a") as f:
            f.write(project_name+"\n")
#     option.append(str(project_name))
    uploaded_files = st.file_uploader("Uploaded files", accept_multiple_files=True, type = ['pdf', 'docx', 'txt'])

    chunks_st = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        st.success(f"Processing: {file_path}")

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
#         chunks_st = []
        if os.path.exists(file_path):
            doc = textract.process(file_path)

            with open('attention_is_all_you_need.txt', 'w') as f:
                f.write(doc.decode('utf-8'))

            with open('attention_is_all_you_need.txt', 'r') as f:
                text = f.read()

            # Step 3: Create function to count tokens
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

            def count_tokens(text: str) -> int:
                return len(tokenizer.encode(text))

            # Step 4: Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                # Set a really small chunk size, just to show.
                chunk_size = 512,
                chunk_overlap  = 100,
                length_function = count_tokens,
            )

            chunks = text_splitter.create_documents([text])
            for i in chunks:
                chunks_st.append(i.page_content)
    #         print("chunks_st[:1]",chunks_st[:1])
    #         print()

        db = FAISS.from_texts(chunks_st[:1], embeddings)
        for i in range(1,len(chunks_st),16):
            db.add_texts(chunks_st[i:i+16])

        db.save_local(project_name)

        
#         text = textract.process(file_path)
#         return text.decode('utf-8')
#     else:
#         return "File does not exist or there was an issue processing the uploaded file."



# def create_embeddings():
#     option=[]
#     st.markdown("## New Project")
#     st.title("Enter Project Details")
#     project_name = st.text_input("Enter Project Name")
#     print("project_name",project_name)
#     if project_name:
#         with open("csv.txt","a") as f:
#             f.write(project_name+"\n")
#     option.append(str(project_name))
#     uploaded_file = st.file_uploader("Uploaded files", accept_multiple_files=True, type = ['pdf', 'docx', 'txt'])
#     process_uploaded_file(uploaded_file, project_name)
        
    
    
#     files = []
#     if len(uploaded_file) != 0:
#         for names in uploaded_file:
#             fgh = process_uploaded_file(names)
#             files.append(fgh)
# #     files = file_path
#         chunks_st = []
#         for file in files:
#         # Step 2: Save to .txt and reopen (helps prevent issues)
#             doc = textract.process(file)
            
#             with open('attention_is_all_you_need.txt', 'w') as f:
#                 f.write(doc.decode('utf-8'))

#             with open('attention_is_all_you_need.txt', 'r') as f:
#                 text = f.read()

#             # Step 3: Create function to count tokens
#             tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

#             def count_tokens(text: str) -> int:
#                 return len(tokenizer.encode(text))

#             # Step 4: Split text into chunks
#             text_splitter = RecursiveCharacterTextSplitter(
#                 # Set a really small chunk size, just to show.
#                 chunk_size = 512,
#                 chunk_overlap  = 100,
#                 length_function = count_tokens,
#             )

#             chunks = text_splitter.create_documents([text])
#             for i in chunks:
#                 chunks_st.append(i.page_content)
#         print("chunks_st[:1]",chunks_st[:1])
#         print()
#         db = FAISS.from_texts(chunks_st[:1], embeddings)
#         for i in range(1,len(chunks_st),16):
#             db.add_texts(chunks_st[i:i+16])
            
#         db.save_local(project_name)


def feedback():
    st.title("Feedback")
    feedback_text = st.text_area("Enter your feedback here", height=200)
    if st.button("Submit Feedback"):
        # Process the feedback (you can add your feedback submission logic here)
        st.write("### Feedback Submitted")
        st.write(feedback_text)
        
        
def load_feedback():
    conn = sqlite3.connect('your_database.db')  # Replace with your database connection
    query = "SELECT * FROM feedback_table"  # Replace with your table name
    feedback_data = pd.read_sql(query, conn)
    conn.close()
    return feedback_data

# Display the feedback in the Streamlit app
def show_feedback(feedback_data):
    st.title("User Feedback")
    st.write(feedback_data)

def start_chat():    
    print("abc")
st.set_page_config(page_title="Azure OPENAI ChatBot", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: red; font-family: "Arial Black", Gadget, sans-serif;'>Welcome to Azure OPENAI ChatBot</h1>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

# option = []
count=0
with col1:
    count+=1
    print(count)
#     st.markdown("## New Project")
#     st.title("Enter Project Details")
#     project_name = st.text_input("Enter Project Name")
#     option.append(option)
#     uploaded_file = st.file_uploader("Uploaded files", accept_multiple_files=True, type = ['pdf', 'docx', 'txt'])
#     files = ['/home/abdullah/Downloads/Train Model for Infralastic Product.docx']
#     if len(uploaded_file) != 0:
#         for names in uploaded_file:
            
    option=create_embeddings()
    print("embed option",option)
#     db.save_local(project_name)
    
with col2:
    st.markdown("## Start a Chat")
    if st.button("Start Chat", key="chat_button"):  # Changed label and removed direct reference to 'Chat'
        st.components.v1.iframe("http://192.168.18.4:8502", height=1000)
#         start_chat()

# with col3:
#     st.markdown("## Give Feedback")
#     feedback()
    
with col3:
    st.markdown('## Get Feedbacks')
    if st.button("Load Feedbacks"):
        feedback_data = load_feedback()
        show_feedback(feedback_data)

