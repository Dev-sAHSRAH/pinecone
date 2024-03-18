import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from pinecone import Pinecone
from pypdf import PdfReader
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain import HuggingFaceHub


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


# function to push data to vector store
def push_to_vector_store(pinecone_index_name,embeddings,docs):
    print("Done!")
    PineconeVectorStore.from_documents(docs,embeddings,index_name = pinecone_index_name)


def pull_from_vector_store(pinecone_index_name,embeddings):
    index = PineconeVectorStore.from_existing_index(pinecone_index_name,embeddings)
    return index

def similar_docs(query,k,pinecone_index_name,embeddings,unique_id):
    index = pull_from_vector_store(pinecone_index_name,embeddings)
    similar_docs = index.similarity_search_with_score(query,int(k),{"unique_id":unique_id})

    # print(similar_docs)
    return similar_docs