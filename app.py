import streamlit as st
from streamlit_chat import message
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from PIL import Image
import os


db_name="rewardola_data"
db_user = "root"
db_password = "root"
db_host = "localhost"


load_dotenv()
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
# db = SQLDatabase.from_uri("sqlite:///rewardola_db (3).sqlite")
gemini_api_key = os.getenv("gemini_api_key")
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key,convert_system_message_to_human=True, temperature=0.0)
agent_executor = create_sql_agent(llm, db=db, verbose=True)

def initialize_session_state():
    st.session_state.setdefault('user_queries', [])
    st.session_state.setdefault('ai_responses', [])

def display_chat():
    reply_container = st.container()
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Question", placeholder="Ask Question About Your Data.", key='input')
        submit_button = st.form_submit_button(label='Send ⬆️')

    if submit_button and user_input:
        generate_response(user_input)

    display_generated_responses(reply_container)



def generate_response(user_input):
    response = agent_executor.invoke({"input": user_input})
    output = response['output']
    
    # Custom formatting for the response to make it a proper sentence
    # Step 1: Strip leading and trailing whitespace
    formatted_output = output.strip()
    
    # Step 2: Capitalize the first letter
    if formatted_output:
        formatted_output = formatted_output[0].upper() + formatted_output[1:]
    
    # Step 3: Ensure proper punctuation at the end (simple example)
    if not formatted_output.endswith('.'):
        formatted_output += '.'
    
    st.session_state['user_queries'].append(user_input)
    st.session_state['ai_responses'].append(formatted_output)




def display_generated_responses(container):
    with container:
        for user_query, ai_response in zip(st.session_state['user_queries'], st.session_state['ai_responses']):
            message(user_query, is_user=True, avatar_style="adventurer") 
            message(ai_response, avatar_style="bottts") 

def main():
    initialize_session_state()
    
    st.title("Genie")

    image = Image.open('chatbot.jpg')
    st.image(image, width=150)
    
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>

            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    display_chat()


if __name__ == "__main__":
    main()
