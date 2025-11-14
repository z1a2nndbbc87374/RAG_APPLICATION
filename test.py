import streamlit as st

st.title("Secret Test")

api_key = st.secrets.get("OPENROUTER_API_KEY")
if api_key:
    st.success("Secret loaded successfully ✅")
    st.write(f"First 5 chars: {api_key[:5]}****")
else:
    st.error("Secret NOT found ❌")
