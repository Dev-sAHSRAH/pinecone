import streamlit as st
import uuid
from utils import *

# create session variable
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] = ''

def main():
    st.set_page_config(page_title = "Student-Recruiter Model")
    st.title("Student-Recruiter Model")
    

    job_desc = st.text_area ("Please paste the Job description Here...",key = "1")
    doc_count = st.text_input("No. of Resumes to return",key="2")

    # Upload pdfs
    pdfs = st.file_uploader("Upload Resumes (only PDF files are allowed)",type= ["pdf"],accept_multiple_files=True)

    submit = st.button("Submit")

    if submit:
        with st.spinner("Fetching..."):
            st.write("Our Process")

            # Creating a unique ID, so that we can use it to query
            # and get only the user uploaded docs from PINECONE vector store
            st.session_state['unique_id'] = uuid.uuid4().hex
            
            st.write(st.session_state['unique_id'])



            # create documents list
            docs = create_doc(pdfs,st.session_state['unique_id'])
            #st.write(docs)
            

            # count of resumes
            st.write(len(docs))

            # creating embeddings instance
            embeddings = create_embeddings_load_data()

            # Push data to Pinecone
            push_to_vector_store("test",embeddings,docs)
            
            # Fetch relevant docs from Pinecone
            relavant_docs = similar_docs(job_desc,doc_count,"test",embeddings,st.session_state['unique_id'])

            st.write(relavant_docs) 


        st.success("Hope I saved your time 🚀")    

# invoking main function
if __name__ == '__main__':
    main()






