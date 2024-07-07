from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv(Path(__file__).parent / ".env")


def text_formatter(retriever_output):
    raw_text = retriever_output[0].page_content
    return raw_text.replace("\n", "")


def create_retirever(dir_ref, embeddings, k=1):
    loader = DirectoryLoader(dir_ref, glob="***/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever


def main():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-small",
        openai_api_version="2023-05-15"
    )
    llm = AzureChatOpenAI(
        model="gpt-4o",
        openai_api_version="2024-02-01",
        temperature=0
    )

    retriever = create_retirever("ref", embeddings, k=1)

    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="以下を参照して、質問に答えてください。\n\n{context}\n\n質問: {question}"
    )

    chain = (
        {"question": RunnablePassthrough(), "context": retriever | text_formatter}
        | prompt_template
        | llm
        | StrOutputParser()  # LLMの出力(AIMessage)を文字列に変換
    )

    print(chain.invoke("賞与は年何回支給されますか？"))


if __name__ == "__main__":
    main()
