import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

load_dotenv()

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = PineconeVectorStore(
    index_name=os.environ['PINECONE_INDEX_NAME'],
    embedding=embeddings
)

#initialize chat model
model = init_chat_model(
    model="qwen2.5:3b",
    model_provider="ollama"
)

@tool(response_format="content_and_artifact")
def retrieve_context(query:str):
    """Retrieve relevant documentation to help answer user queries about Langchain"""
    
    retrieved_docs = vectorstore.as_retriever().invoke(query,k=4)
    
    # serialize for llm
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get("source","unknown")} \n\nContent: {doc.page_content}") 
        for doc in retrieved_docs
    )
    
    return serialized, retrieved_docs
    

def run_llm(query: str) -> Dict[str,Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation

    Args:
        query (str): The users question

    Returns:
        Dict[str,Any]: 
            - answer: The generated answer
            - context: List of retrieved documents
    """
    
    # create the agent with the retrieval tool
    
    system_prompt = (
        "You are a helpful AI assistant that answers questions about Langchain documentation."
        "You have access to a tool that retrieves relevant documentation."
        "Use the tool to find relevant information before answering the questions."
        "Always cite the sources you use in your answers."
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    
    # build messages list
    messages = [{"role":"user","content":query}]
    
    # invoking the agent
    response = agent.invoke({"messages": messages})
    
    # extract the message from the last ai message
    answer = response['messages'][-1].content
    
    # extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response['messages']:
        # check if this is a ToolMessage with artifact
        if isinstance(message,ToolMessage) and hasattr(message,"artifact"):

            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {
        "answer": answer,
        "context": context_docs
    }


if __name__ == "__main__":
    result = run_llm(query="What are deep agents ?")
    print(result)