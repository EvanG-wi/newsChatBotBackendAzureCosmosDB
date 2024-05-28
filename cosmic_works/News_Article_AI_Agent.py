import os
import requests 
import json
from openai import AzureOpenAI
from pymongo import MongoClient, UpdateOne
import time
from dotenv import load_dotenv

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch
from langchain.schema.document import Document
from langchain.agents import Tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.tools import StructuredTool
from langchain_core.messages import SystemMessage
from typing import List


load_dotenv(".env")
CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
ENDPOINT = "https://polite-ground-030dc3103.4.azurestaticapps.net/api/v1"
API_VERSION = "2024-02-01"
EMBEDDING_MODEL = "text-embedding-3-small"
API_KEY = os.environ.get("AOAI_KEY")

MODEL_NAME = 'gpt-35-turbo' #'gpt-4'
collection = MongoClient(CONNECTION_STRING).newspapers['newsArticles']

class News_Article_AI_Agent:
    def __init__(self,session_id: str):
        llm = AzureChatOpenAI(
            temperature=.8,
            openai_api_version = API_VERSION,
            azure_endpoint = ENDPOINT,
            openai_api_key = API_KEY,
            azure_deployment = MODEL_NAME
        )
        self.ai_client = AzureOpenAI(
            azure_endpoint=ENDPOINT,
            api_key=API_KEY,
            api_version=API_VERSION
        )
        self.embedding_model = AzureOpenAIEmbeddings(
            openai_api_version = API_VERSION,
            azure_endpoint = ENDPOINT,
            openai_api_key = API_KEY,
            azure_deployment = EMBEDDING_MODEL,
            chunk_size = 10
        )

        system_message = SystemMessage(content="""Pretend you are a person living in 1945, 
                                                        use the information in any provided newspaper articles to answer 
                                                        the user's questions in a conversational style try to use at least one quote""")

        self.agent_executor = create_conversational_retrieval_agent(
            llm,
            self.__create_agent_tools(),
            system_message = system_message,
            memory_key = session_id,
            verbose=True
            )
        
    def run(self,prompt :str) -> str:
        result = self.agent_executor({'input':prompt})
        return result['output']+'*'
    
    def __create_news_article_vector_store_retriever(self, collection_name='newsArticles', top_k = 2):
        vector_store = AzureCosmosDBVectorSearch.from_connection_string(
            connection_string=CONNECTION_STRING,
            namespace = "newspapers.newsArticles",
            embedding = self.embedding_model,
            index_name = "VectorSearchIndex",
            embedding_key = "contentVector",
            text_key = "_id"
            )
        return vector_store.as_retriever(search_kwargs= {"k":top_k})
    
    def __create_agent_tools(self) -> List[Tool]:                                                 
        search_retriever = self.__create_news_article_vector_store_retriever()
        search_retriever_chain = search_retriever | format_docs
        tools = [Tool(name = "article_search",
                      func = search_retriever_chain.invoke,
                      description="Whenever it would be helpful, search for an article about the subject. input is one string describing what information the article should contain")
                 ]
        return tools
        
    #def generate_embeddings(self,query: str):
        #response= self.ai_client.embeddings.create(input=query,model=EMBEDDING_MODEL)
        #embeddings = response.data[0].embedding
        ##time.sleep(.5)
        #return embeddings    
    
    #def vector_search(self,query, num_results):
        #embedded_query = self.generate_embeddings(query)
        #pipeline =[
        #{'$search' : #search best vectors 
          #{ "cosmosSearch": 
                      #{
                          #"vector" : embedded_query,
                          #"path" : "contentVector",
                          #"k" : num_results
                          #},
                      #"returnStoredSource" : True
                      #}
        #},
         #{ "$project" : 
           #{              #only return relevant parts
               #'date' : 1,
               #'headline' : 1,
               #'article' : 1,
               #}
           #}
         #]
        #return collection.aggregate(pipeline)
    
    #def rag_with_vector_search(self,query: str, num_results = 1) -> str:
        #results_cursor = self.vector_search(query,num_results)
        #articles_prompt_injection = ""
        #for article in results_cursor:
            #articles_prompt_injection += json.dumps(article,default=str)+'\n\n'
        
        #completion = self.ai_client.chat.completions.create(
            #model = MODEL_NAME,
            #self.messages.append(
                #{'role' : 'user', 'content' : f"article:\n{articles_prompt_injection} \n question: {query}"}
                
            #)
        #return completion.choices[0].message.content
          
def format_docs(docs:List[Document]) -> str:
    """
    Prepares the product list for the system prompt.
    """
    str_docs = []
    for doc in docs:
        # Build the product document without the contentVector
        doc_dict = {"_id": doc.page_content}
        doc_dict.update(doc.metadata)
        if "contentVector" in doc_dict:
            del doc_dict["contentVector"]
        str_docs.append(json.dumps(doc_dict, default=str))
    # Return a single string containing each product JSON representation
    # separated by two newlines
    return "\n\n".join(str_docs)   

print('backend_agent initalized')
#goose = News_Article_AI_Agent('honk')
#print(goose.run("Tell me about new technological discoveries that are changing peoples day to day lives"))
#print(goose.run("Can you explain the context of the quote more?"))
#print(goose.run("what can you tell me about how china is doing?"))