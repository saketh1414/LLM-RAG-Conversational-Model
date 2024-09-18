import os
import streamlit as st
import requests

#add your flask_app file location before \faiss_index.bin
faiss_index_file = r"D:\webscrap conversational\llm_search_template (2)\llm_search_template\flask_app\faiss_index.bin"

st.title("LLM-based RAG Search-conversational model")
# Input for user query
st.write("Information about your First query will be webscrapped and used for further discussion")
st.write("enter \"exit\" to start fresh and webscrap about new topic - if you get error 500 try typing \"exit\"")

query = st.text_input("Enter your query:")

if st.button("Search"):
    
    if query.lower() == 'exit':
        if os.path.exists(faiss_index_file):
            os.remove(faiss_index_file)
            st.write("webscrpped data is deleted")
        else:
            st.write("we didn't webscrapped yet!")
    else:
        #st.write(f"The entered query is:  {query}") 

        #this is the url of flask api
        FlaskUrl="http://localhost:5001/query"

        try:
            #calling flask api to get response
            #sending a POST request and data is sent through json format(basically dictionary)
            response = requests.post(FlaskUrl,json={"query":query})
            
            if response.status_code == 200:
                # Display the generated answer
                final_result = response.json().get('result', "No answer received.")
                st.write("Result:", final_result)
            else:
                st.error(f"Error: {response.status_code}")
        except Exception as ex:
            st.error(f"failed to connect to backend: {ex}")




            