import os
import requests
from bs4 import BeautifulSoup
import re
import faiss
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")


### FUNCTIONS RELATED TO WEBSCRAPPING

def search_articles(query):  #returns content from different websites.serper is used to generate urls.
    headers={'X-API-KEY':SERPER_API_KEY}
    parameters={'q':query,'num':5} #here we can give parameter 'num':5 if we want to fetch details of only 5 websites,default is 10
    
    response=requests.get('https://google.serper.dev/search',headers=headers,params=parameters)
    searchResults=response.json()

    articles = []
    #generally all the details like link,content is present in organic key of serper response.
    for item in searchResults.get('organic',[]):
        url=item.get('link')
        #title=item.get('title')  
        article_content=fetch_article_content(url)
        articles.append({'url':url,'content':article_content})
        print("serper data should be producted here")
    return articles

def fetch_article_content(url):  #used to webscrap the information present in url
    content = ""
    print("inside fetch article beautiful soup")
    try:
        response=requests.get(url)
        soup_data=BeautifulSoup(response.content,'html.parser')
        # implementation of fetching headings and content from the articles
        for heading in soup_data.find_all(['h1','h2','h3']):
            content+=heading.get_text()+"\n"
        for para in soup_data.find_all('p'):
            content+=para.get_text()+"\n"
    except Exception as ex:
        print(ex)

    return content.strip()


def concatenate_content(articles):  
    full_text = ""
    for article in articles:
        full_text += clean_text(article['content']) + '\n'
    print('inside clean and concatenate content')
    return full_text

def clean_text(text):   #used to clean the text.
    # Remove HTML tags
    text = re.sub(r'<[^>]*?>', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    # Trim leading and trailing whitespace
    text = text.strip()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text



### FUNCTIONS RELATED TO FAISS DATABASE

model = SentenceTransformer('all-MiniLM-L6-v2')
vectordb_file_path="faiss_index.bin"

def store_in_faiss(total_content):  #used to store webscrapped content in faiss database
    if not isinstance(total_content, list) or not all(isinstance(item, str) for item in total_content):
        raise TypeError("total_content must be a list of strings.")
    embeddings = model.encode(total_content, convert_to_numpy=True)
    dimension = embeddings.shape[1]  # The dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance for indexing
    index.add(embeddings)
    faiss.write_index(index, vectordb_file_path)


def retrieve_from_faiss(query,total_content):   #used to retrieve related data from whole webscrap based on the query
    index = faiss.read_index(vectordb_file_path)
    query_embedding = model.encode(query, convert_to_numpy=True).reshape(1, -1)
    D, I = index.search(query_embedding, 20)  #it returns indices of total_content which are top results
    retrieved_texts = []
    for i in I[0]:
        if i != -1:  # Check for valid index
            retrieved_texts.append(total_content[i])  # total_content should be in scope
    #print("retrieved from faiss:\n",retrieved_texts)
    return retrieved_texts


def is_faiss_index_empty(index_file='faiss_index.bin'):  #checks if faiss DB exists or any data is present in it.
    if not os.path.isfile(vectordb_file_path):
        return True
    index = faiss.read_index(vectordb_file_path)
    return index.ntotal == 0



### FUNCTIONS RELATED TO LLAMA MODEL
#we can also use gpt-4 here by making small changes.
llm = ChatGroq(
        model_name="llama-3.1-70b-versatile",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )
first_query=""
content_in_list=[]

def generate_list_content(total_content,query):  #used to generate list of many important sentences from whole webscrapped content. 
    prompt_input = PromptTemplate.from_template(
        """
        For the query:{query},we had webscrapped content from 5 different websites,
        now return a string containing multiple sentences which are meaningful and covers all the different topics from different websites.
        this string will be input for a vector database used to retrieve data based on specific topic in the whole content.
        i dont want any metadata or other parameters in it.i only want a long string of sentences in which all the topics are covered.
        you have to only generate the string format and return nothing else.generate atleast 200 meaningful sentences .dont include any \\n and dont use "" or '' for highlighting in output.
        the webscrapped content is:\n{total_content}\n 
        only return one string containing sentences  to directly give input to vector database.
        example output:
        "this is sentence 1.this is sentence 2.this is sentence 3"
        ### String (NO PREAMBLE):
        """
    )
    chain_generator = prompt_input | llm
    response = chain_generator.invoke({"total_content": total_content,"query": query})
    
    sentence_list = [sentence.strip() for sentence in response.content.split(".") if sentence.strip()]
    #print("generate_list_content op:\n",sentence_list)
    return sentence_list


def generate_answer(first_query,content,history,query):  #used to generate response based on the query and related data retrieved from DB.
                                                         #conversation history is also passed.
    prompt_input = PromptTemplate.from_template(
        """
        for the first query entered by the user:{first_query} some information is webscrapped and stored in database,
        now for query: {query}, the database returned the content:{content}.
        if this content and {first_query} is any way related to {query} use this content.\n\n
        the below chat was already done with you and the user:
        \n{history}\n
        give reply according to the query below:\n{query}.
        do not mention that based on previous converstions or data retrieved from databases, just give the answer to {query}.
        """
    )
    chain_generator = prompt_input | llm
    response = chain_generator.invoke({"first_query":first_query,"content": content,"history":"\n".join([msg.content for msg in history]), "query": query})
    return str(response.content)
