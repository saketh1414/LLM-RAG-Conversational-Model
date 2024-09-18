
from flask import Flask, request,jsonify
import os
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import utils

app = Flask(__name__)
memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)


@app.route('/query', methods=['POST'])
def query():
    
    data=request.get_json()   # get the data from streamlit app
    query=data.get('query',"") #second one is default parameter
    
    if not query:
        return jsonify({'error':'no query is given'}),400  # status code 400 is known as Bad Request.
    

    existing_memory = memory.load_memory_variables({})
    conversation_history = existing_memory.get('chat_history', [])
    if isinstance(conversation_history, str):
        conversation_history = [conversation_history]

    print("is fiass database empty:\n",utils.is_faiss_index_empty('faiss_index.bin'))
    if utils.is_faiss_index_empty('faiss_index.bin'):
        articles= utils.search_articles(query)
        total_content=utils.concatenate_content(articles)
        utils.content_in_list=utils.generate_list_content(total_content,query)
        #print("type :" ,type(content_in_list[0]),"type of out:",type(content_in_list))
        #print("json from llama:\n",content_in_list)
    
        utils.store_in_faiss(utils.content_in_list)
        utils.first_query=query
        #print("the content list is: ",utils.content_in_list)

    #conversation_history.append(HumanMessage(content=query))
    print("Main topic is: ",utils.first_query)
    #print("the content list is: ",utils.content_in_list)

    
    result = utils.generate_answer(utils.first_query,utils.retrieve_from_faiss(query,utils.content_in_list), conversation_history, query)

    conversation_history.append(HumanMessage(content=query))
    conversation_history.append(AIMessage(content=result))
    #print("conversation history:\n",conversation_history)
    print('\nhuman:\n',query,"\nAi model:\n",result,"\n")

    # return the jsonified text back to streamlit
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='localhost', port=5001)

