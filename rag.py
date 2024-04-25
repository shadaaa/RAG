# import all necessary libraries
import pickle
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

#load the pdf you need
#to split the text into chunks use split_documents.
loader5 = PyPDFLoader("C:/Users/ASUS/OneDrive/Desktop/dataset/resome.pdf")
docs5 = loader5.load()
text_splitter = CharacterTextSplitter(chunk_size=500)
split_docs5 = text_splitter.split_documents(docs5)
#embed your splitted chunks
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key="your_api_key")
vectordb4 = FAISS.from_documents(split_docs5, embeddings)

# to save locally
directory_path = "C:/Users/ASUS/OneDrive/Desktop/hr/fulfillment/functions/rag"
index_file_path = os.path.join(directory_path)
vectordb4.save_local(index_file_path)
new_db = FAISS.load_local(index_file_path, embeddings, allow_dangerous_deserialization=True)

# to retrieve the answer relevant to the user query
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
retriever = vectordb4.as_retriever(search_type="similarity")
rqa = RetrievalQA.from_chain_type(llm=OpenAI( openai_api_key="your_api_key"),
                                  chain_type="stuff",
                                  retriever=retriever,
                                 return_source_documents=True)
#ask the question
def response(llm_response):
    result = llm_response['result']
    print(result)

while True:
    user_input=input(f"Input Prompt:")
    if user_input == 'exit':
        print("Exiting")
        break
    if user_input=="":
        continue
    result=rqa({'query':user_input})
    response(result)



