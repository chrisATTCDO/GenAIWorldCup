import streamlit as st  
import pandas as pd  
import json  
from api import score_model  
import time  
from datetime import datetime  
  
def main():  
    st.title("Wallstreet Guru \n Team : AT&T CDO - Databricks Hackathon")  
  
    # Initialize session state for chat sessions  
    if 'chat_sessions' not in st.session_state:  
        st.session_state['chat_sessions'] = {}  
    if 'selected_chat' not in st.session_state:  
        st.session_state['selected_chat'] = None  
  
    # Sidebar for managing chat sessions  
    st.sidebar.header("Chat Sessions")  
    chat_name = st.sidebar.text_input("Enter new chat name:")  
    if st.sidebar.button("Create Chat"):  
        if chat_name:  
            if chat_name not in st.session_state['chat_sessions']:  
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
                st.session_state['chat_sessions'][chat_name] = {  
                    "created_at": timestamp, "history": [], "query": "", "response": "", "citations": []  
                }  
                st.session_state['selected_chat'] = chat_name  
        else:  
            st.sidebar.warning("Please enter a chat session name.")  
  
    # Display existing chat sessions in sidebar  
    selected_chat = st.sidebar.selectbox(  
        "Select a chat session:",  
        list(st.session_state['chat_sessions'].keys()),  
        index=list(st.session_state['chat_sessions'].keys()).index(st.session_state['selected_chat']) if st.session_state['selected_chat'] else 0  
    )  
    st.session_state['selected_chat'] = selected_chat  
  
    # Main panel for chat interaction  
    if selected_chat:  
        st.header(f"Chat: {selected_chat}")  
        chat_data = st.session_state['chat_sessions'][selected_chat]  
        chat_history = chat_data["history"]  
  
        # Display chat history  
        for entry in chat_history:  
            st.markdown(f"**User Query:** {entry['query']}")  
            st.markdown(f"**Assistant Response:** {entry['answer']}")  
            if 'citations' in entry:  
                with st.expander("Show Citations", expanded=False):  
                    for citation in entry['citations']:  
                        st.markdown(citation.get("content", ""), unsafe_allow_html=True)  
  
        # Use a form for the new query input  
        with st.form(key='query_form'):  
            query = st.text_input("Enter your question:", key=f"query_{selected_chat}")  
            submit_button = st.form_submit_button("Submit Query")  
  
        if submit_button and query:  
            chat_data['query'] = query  
  
    # Placeholder for dynamic content  
    response_placeholder = st.empty()  
  
    # Process and update chat history if there's a new query  
    if selected_chat and chat_data['query']:  
        input_df = pd.DataFrame({  
            "query": [chat_data['query']],  
            "chat_history": json.dumps(chat_history)  
        })  
  
        with st.spinner("Processing your request..."):  
            progress_bar = st.progress(0)  
            for percent_complete in range(100):  
                time.sleep(0.01)  
                progress_bar.progress(percent_complete + 1)  
  
            result = score_model(input_df)  
  
        # Store results in session state  
        chat_data['response'] = result.get("predictions", {}).get("response", "")  
        chat_data['citations'] = result.get("predictions", {}).get("citations", [])  
  
        chat_history.append({  
            "query": chat_data['query'],  
            "answer": chat_data['response'],  
            "citations": chat_data['citations']  
        })  
  
        # Clear the query to avoid reprocessing  
        chat_data['query'] = ""  
  
    # Display the latest response  
    if selected_chat and chat_data['response']:  
        response_placeholder.markdown(f"**Assistant Response:** {chat_data['response']}")  
        if chat_data['citations']:  
            with st.expander("Show Citations", expanded=False):  
                for citation in chat_data['citations']:  
                    st.markdown(citation.get("content", ""), unsafe_allow_html=True)  
  
if __name__ == "__main__":  
    main()  

