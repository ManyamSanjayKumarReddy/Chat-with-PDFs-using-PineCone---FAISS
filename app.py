import streamlit as st
import os
import time  # To introduce delays for the typing effect
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Pinecone instance
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)  # Instantiate Pinecone with API key

# Define Pinecone index parameters
INDEX_NAME = "chat-memory"
DIMENSION = 768  # Ensure this matches the embeddings model's output dimension

# Create or connect to a Pinecone index with serverless spec
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric='cosine',  # Change to your preferred metric (e.g., cosine, dot_product)
        spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Adjust cloud and region as needed
    )

# Access the index for operations
pinecone_index = pc.Index(INDEX_NAME)

# Function to store chat summary in Pinecone
def store_summary_in_pinecone(question, summary):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector = embeddings.embed_query(summary)

    # Store the embedding in Pinecone
    pinecone_index.upsert([{"id": str(hash(question)), "values": vector, "metadata": {"question": question, "summary": summary}}])

# Function to create the conversational AI chain
def get_conversational_chain():
    prompt_template = """
    You are an Exclusive Breast Cancer Professor and Researcher. Answer all the questions in detail, related to Breast Cancer,
    leveraging your expertise. Remember previous conversations to maintain context. You are always welcome to give medical advice based on Latest data you have.

    Context:\n {context}?\n
    Question:\n {question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

# Function to handle chat interaction and store the summary
def handle_chat_and_store_summary(user_question):
    chain = get_conversational_chain()
    response = chain({"input_documents": [], "question": user_question}, return_only_outputs=True)

    # Summarize the response for memory storage
    summary = f"Q: {user_question}\nA: {response['output_text']}"
    store_summary_in_pinecone(user_question, summary)  # Store in Pinecone

    return response["output_text"]

# Function to simulate typing effect for AI response
def display_typing_effect(response_text):
    typing_container = st.empty()  # Use st.empty to update the content dynamically
    display_text = ""
    
    for char in response_text:
        display_text += char
        typing_container.markdown(f"**AI:** {display_text}")  # Update text incrementally
        time.sleep(0.0009)  # Adjust the delay to make typing effect slower or faster

# Main Streamlit app function with chatbox interface
def main():
    st.set_page_config("Memory-Based Chat Interface", layout="wide")
    st.title("AI-Powered Chat with Memory - Breast Cancer Expert 💁‍♂️")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # To keep track of conversation history

    # Display chat history in a chatbox-like interface
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state["chat_history"]:
            if chat["type"] == "user":
                st.markdown(f"**You:** {chat['message']}")
            else:
                st.markdown(f"**AI:** {chat['message']}")

    # Input field at the bottom for user input
    user_input = st.text_input("Type your message and press Enter", key="user_input")

    if user_input:
        # Add user message to chat history
        st.session_state["chat_history"].append({"type": "user", "message": user_input})

        # Handle the chat interaction
        chat_response = handle_chat_and_store_summary(user_input)

        # Display AI's response with typing effect
        st.session_state["chat_history"].append({"type": "ai", "message": chat_response})  # Store full response in history
        display_typing_effect(chat_response)  # Show typing effect for the response

if __name__ == "__main__":
    main()
 