#!/usr/bin/env python
# coding: utf-8

# In[1]:


#PyPDF2 is used to read the pdf file which is given.FAISS is used to store the vector which is generated after embedding the text.
get_ipython().system('pip install langchain')
get_ipython().system('pip install OpenAi')
get_ipython().system('pip install PyPDF2')
get_ipython().system('pip install faiss-cpu')
#!pip install tiktoken


# In[2]:


#elasticvectorsearch is used to identify the meaning of unstructured data.here not used.
#pinecone is vector database for high dim vector.not used

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
#from langchain.vectorstores import ElasticVectorSearch,pinecone,weaviate,FAISS


# In[3]:


import os
os.environ["OPEN_API_KEY"] = "api"


# In[6]:


reader = PdfReader("C:/Users/ASUS/OneDrive/Desktop/dataset/ragdoc.pdf")
reader


# In[7]:


#using PdfReader the pdf is stored.now to obtain text we use a for loop.
#each page is extracted and from pages text and stored into doc
doc = ""
for i,page in enumerate(reader.pages):
    word = page.extract_text()
    if word:
        doc += word


# In[8]:


doc


# In[9]:


from langchain.text_splitter import CharacterTextSplitter


# In[10]:


#splitter is used to split the text into chunks which is specified in the function.
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(doc)


# In[11]:


texts


# In[12]:


len(texts)


# In[13]:


texts[1]


# In[ ]:





# In[14]:


get_ipython().run_line_magic('env', 'OPENAI_API_KEY=api')


# In[17]:


embeddings =OpenAIEmbeddings()


# In[18]:


#in the variable search,the texts with corresponding embedding is stored
from langchain_community.vectorstores import FAISS
search = FAISS.from_texts(texts,embeddings)


# In[ ]:





# In[19]:


from langchain_core.documents import Document


# In[20]:


from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


# In[21]:


#a chain is group of words or texts.it is loaded into a chain variable.
#stuff will take document,pass into prompt then give to llm.
#it's used for small number of docs
chain = load_qa_chain(OpenAI(),chain_type = "stuff")


# In[22]:


#the similarity search is done using embeddings of the text and compared with query.
#run the chain by giving the docs and query
query = "What are key elements of Generative?"
docs = search.similarity_search(query)
chain.run(input_documents=docs,question=query)


# In[23]:


query = "What is Generative AI?"
docs = search.similarity_search(query)
chain.run(input_documents=docs,question=query)


# In[24]:


query = "What is a LLM model?"
docs = search.similarity_search(query)
chain.run(input_documents=docs,question = query)


# In[25]:


query = "What is chain in a LLM model?"
docs = search.similarity_search(query)
chain.run(input_documents=docs,question=query)


# In[26]:


query = "What is rag?"
docs = search.similarity_search(query)
chain.run(input_documents=docs,question=query)


# In[ ]:


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(openai_api_key="api")


# In[ ]:


llm.invoke("how people writes?")


# In[ ]:


from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system","You are worlds best seller books author"),
    ("user","{input}")
])


# In[ ]:


chain = prompt|llm


# In[ ]:


chain.invoke({"input": "how people writes?"})


# In[ ]:


from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()


# In[ ]:


chain = prompt|llm|output_parser


# In[ ]:


chain.invoke({"input": "how people writes?"})


# # RETREIVAL
# 

# In[ ]:


from langchain.chains.combine_documents import create_stuff_documents_chain
prompt = ChatPromptTemplate.from_template("""Answer the following question based on the context:
<context>
{context}
</context>
Question:{input}""")
doc_chain = create_stuff_documents_chain(llm,prompt)


# In[ ]:


from langchain_core.documents import Document
doc_chain.invoke({
    "input": "What is RAG?",
    "context": [Document(page_content="RAG is a retreival model.")]
})


# In[ ]:


from langchain.chains import create_retrieval_chain
retriever = search.as_retriever()
retrieval_chain = create_retrieval_chain(retriever,doc_chain)


# In[ ]:


response = retrieval_chain.invoke({"input":"What is RAG?"})
print(response["answer"])


# # CONVERSATIONAL RETREIVAL CHAIN
# 

# In[ ]:


from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","{input}"),
    ("user","give answer based on above conversation")
])
retriever_chain = create_history_aware_retriever(llm,retriever,prompt)


# In[ ]:


from langchain_core.messages import HumanMessage, AIMessage
chat_history =[HumanMessage(content="what is NLP?"),AIMessage(content="Yes")]
response1 = retriever_chain.invoke({
    "chat_history":chat_history,
    "input":"what are its components?give short answer."
})
print(response1)


# In[ ]:


chat_history.append(HumanMessage(content="what is its full form?"))


# In[ ]:


response2 = retriever_chain.invoke({
    "chat_history":chat_history,
    "input":"what is its full form?"
})
print(response2)


# In[ ]:


chat_history.append(HumanMessage(content="is chatgpt part of it?"))


# In[ ]:


response3 = retriever_chain.invoke({
"chat_history":chat_history,
"input":"is chatgpt part of it?"
})
print(response3)


# In[ ]:





# In[ ]:





# In[ ]:




