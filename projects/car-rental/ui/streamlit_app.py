import streamlit as st
import requests
import uuid

st.title('Car rental booking chatbot')

# User ID input (could be from login, or generated)
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())  # Unique per session

user_id = st.session_state["user_id"]

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.text_input("Ask a question about car rentals:")

if st.button("Send") and user_input:
    response = requests.post(
        "http://localhost:8000/chat",  # Your FastAPI endpoint
        json={"user_id": user_id, "question": user_input}
    ).json()
    st.session_state["chat_history"].append(("You", user_input))
    st.session_state["chat_history"].append(("Bot", response["response"]))

for sender, msg in st.session_state["chat_history"]:
    st.markdown(f"**{sender}:** {msg}")
