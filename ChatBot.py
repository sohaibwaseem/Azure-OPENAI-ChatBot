import textract
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain

os.environ['OPENAI_API_KEY'] = 'a194af7e52e64d22b86f65ac97bc0fd9'
os.environ['OPENAI_API_TYPE'] = 'azure'
os.environ['OPENAI_API_VERSION'] = '2023-09-15-preview'
os.environ['OPENAI_API_BASE'] = "https://vipindemo.openai.azure.com/"

# def Chat(query, db):


    
# #     prompt_template = PromptTemplate(
# #     input_variables=["query", ],
# #     template=prompt
# #     )
    
#     llm = AzureChatOpenAI(
#         deployment_name="vipin-gpt35",
#         model='gpt-3.5-turbo-16k')
    
    
#     template = """
#     {summaries}
#     {question}
#     """

#     chain = RetrievalQAWithSourcesChain.from_chain_type(
#         llm=OpenAI(temperature=0),
#         chain_type="stuff",
#         retriever=docsearch.as_retriever(),
#         chain_type_kwargs={
#             "prompt": PromptTemplate(
#                 template=template,
#                 input_variables=["summaries", "question"],
#             ),
#         },
#     )
    
    
    
#     chain = load_qa_chain(llm, chain_type="stuff", )
#     docs = db.similarity_search(query)
    
#     prompt = f"""Answer the question based on the context below. If the
#     question cannot be answered using the information provided answer
#     with "I don't know".

#      I have supplied you with a set of documents, and your responses should be exclusively based on the content within these files. Please adhere to the following guidelines when generating responses from the provided documents:

#     1. Refrain from introducing additional information independently.
#     2. Utilize the names specified in the files without substituting them with any other names.
#     3. If the requested information is absent from the provided documents, kindly respond with "I don't know".
#     4. Should I initiate the conversation with "Hello" or "Hi," your response should be to inquire, "How may I be of assistance to you?"
#     5. Rely solely on the information contained within the provided documents for your responses.
#     6. Do not consult any other sources for answers, aside from the documents I have furnished to you."

#     Question: {query}
    
#     Answer: """    
    
#     response = chain.run(input_documents=docs, question=query)
#     return response


from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
# import chainlit as cl


# db = FAISS.load_local('/home/abdullah/Documents/Upwork/Pankaj/Honeywell', embeddings)



custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=False,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = AzureChatOpenAI(
deployment_name="vipin-gpt35",
model='gpt-3.5-turbo-16k',
temperature = 0)
    return llm

#QA Model Function
# def qa_bot():
    
#     llm = load_llm()
#     qa_prompt = set_custom_prompt()
#     qa = retrieval_qa_chain(llm, qa_prompt, db)

#     return qa

#output function
def final_result(query, db):
    embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002', 
      deployment='vipin-ada',
       openai_api_key = os.environ['OPENAI_API_KEY'],
       openai_api_type=  os.environ['OPENAI_API_TYPE'],
        openai_api_version= os.environ['OPENAI_API_VERSION'],
        openai_api_base= os.environ['OPENAI_API_BASE'])
#     db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa_result = retrieval_qa_chain(llm, qa_prompt, db)
#     print("qa_result",qa_result)
      
    response = qa_result({'query': query})
    print("response",response)
    return response["result"]

