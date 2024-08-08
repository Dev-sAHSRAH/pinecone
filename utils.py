import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from pinecone import Pinecone,ServerlessSpec
from pypdf import PdfReader
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain import HuggingFaceHub
import os
os.environ['PINECONE_API_KEY'] = "c3862e79-b910-4cf8-9a1a-143b4732bf9c"


#Extract info from PDF
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

# iterate over all files one by one
def create_doc(user_pdf_list,unique_id):
    docs=  []
    for filename in user_pdf_list:
        chunks = get_pdf_text(filename)

        docs.append(Document(
            page_content=chunks,
            metadata = {"name":filename.name,"type":filename.type,"size":filename.size,"unique_id":unique_id},
        ))

    return docs


# create embedding
def create_embeddings_load_data():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


#function to push data to vector store
# def push_to_pinecode(pinecone_apikey,pinecode_environment,pinecone_index_name,embeddings,docs):
#     pc = Pinecone(api_key=pinecone_apikey, environment=pinecode_environment)

#     # Check if the index exists, create it if it doesn't
#     if pinecone_index_name not in pc.list_indexes().names():
#         pc.create_index(
#             name=pinecone_index_name,
#             dimension=1536,  # Adjust dimension as needed
#             metric='euclidean'  # Adjust metric as needed
#         )
#         print("Index created...")
#     else:
#         print("Index already exists")

#     #Push data to Pinecone index
#     PineconeVectorStore.from_documents(docs,embeddings,index_name = pinecone_index_name)


def push_to_vector_store(pinecone_index_name,embeddings,docs):
    print("Done!")
    PineconeVectorStore.from_documents(docs,embeddings,index_name = pinecone_index_name)
    


def pull_from_vector_store(pinecone_index_name,embeddings):
    try:
        index = PineconeVectorStore.from_existing_index(pinecone_index_name, embeddings)
        print("Index retrieved successfully:", index)

        return index
    except Exception as e:
        print("Error retrieving index:", e)
        return None

def similar_docs(query,k,pinecone_index_name,embeddings,unique_id):
    # pc = Pinecone(api_key="c3862e79-b910-4cf8-9a1a-143b4732bf9c")
    # index1 = pc.Index(pinecone_index_name)
    # print(index1)
    # print(index1.describe_index_stats())
    index = pull_from_vector_store(pinecone_index_name, embeddings)
    if index is not None:
        try:
            similar_docs = index.similarity_search_with_score(query, int(k))
            print("Similar documents found:", similar_docs)
            return similar_docs
        except Exception as e:
            print("Error searching for similar documents:", e)
            return []
    else:
        return []

