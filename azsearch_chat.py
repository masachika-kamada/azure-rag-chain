import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv(Path(__file__).parent / ".env")


def initialize_vector_store(index_name, embedding_function):
    return AzureSearch(
        azure_search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.environ.get("AZURE_SEARCH_ADMIN_KEY"),
        index_name=index_name,
        embedding_function=embedding_function
    )


def main():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-large",
        openai_api_version="2023-05-15"
    )
    llm = AzureChatOpenAI(
        model="gpt-4o",
        api_version="2024-02-01",
        temperature=0
    )
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="以下を参照して、質問に答えてください。\n\n{context}\n\n質問: {question}"
    )
    vector_store = initialize_vector_store(
        index_name="diary-vector",
        embedding_function=embeddings.embed_query
    )

    retriever = RunnableLambda(vector_store.similarity_search).bind(k=3, search_type="hybrid")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    user_input = "部署に新人が参加した時のことを忘れてしまったから教えて"
    output = chain.invoke(user_input)

    print(f"human: {user_input}")
    print(f"assistant: {output}")


if __name__ == "__main__":
    main()
