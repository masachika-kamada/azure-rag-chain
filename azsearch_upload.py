import os
from pathlib import Path
from dotenv import load_dotenv
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
)
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv(Path(__file__).parent / ".env")


def load_documents(directory_path):
    loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
    return loader.load()


def setup_index_fields(vector_search_dimensions, analyzer_name="standard.lucene"):
    return [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
            analyzer_name=analyzer_name
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=vector_search_dimensions,
            vector_search_profile_name="myHnswProfile",
        ),
        SearchableField(
            name="metadata",
            type=SearchFieldDataType.String,
            searchable=True,
            analyzer_name=analyzer_name
        ),
    ]


def initialize_vector_store(index_name, embedding_function, fields):
    return AzureSearch(
        azure_search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.environ.get("AZURE_SEARCH_ADMIN_KEY"),
        index_name=index_name,
        embedding_function=embedding_function,
        fields=fields,
    )


def main():
    docs = load_documents("./ref")
    print(f"{len(docs)} documents have been loaded")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-large",
        openai_api_version="2023-05-15"
    )
    fields = setup_index_fields(
        vector_search_dimensions=len(embeddings.embed_query("Text")),
        analyzer_name="ja.microsoft"
    )
    vector_store = initialize_vector_store(
        # index_name="diary-vector",
        index_name="hoge2",
        embedding_function=embeddings.embed_query,
        fields=fields
    )
    vector_store.add_documents(documents=docs)
    print(f"{len(docs)} documents have been added to the AzureSearch index")


if __name__ == "__main__":
    main()
