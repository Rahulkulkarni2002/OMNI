import streamlit as st
import requests
import time

st.set_page_config(page_title="Omni — NVIDIA NIM Assistant", page_icon="🤖")
st.title("Omni")
st.caption("Ask about NVIDIA NIM & Triton deployment, models, and GPU requirements")

API_URL = "http://localhost:8000/ask"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("reasoning"):
            with st.expander("Reasoning steps"):
                for step in msg["reasoning"]:
                    st.markdown(f"**{step['tool']}** — {step['detail']}")
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- [{s['title']}]({s['url']})")

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(API_URL, json={"question": question}, timeout=90)
                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
                sources = data.get("sources", [])
                reasoning = data.get("reasoning", [])
            except Exception as e:
                answer = f"Couldn't reach Omni's backend — make sure main.py is running. ({e})"
                sources = []
                reasoning = []

        if reasoning:
            with st.expander("Reasoning steps", expanded=True):
                for step in reasoning:
                    st.markdown(f"**{step['tool']}** — {step['detail']}")

        def stream_words():
            for word in answer.split(" "):
                yield word + " "
                time.sleep(0.02)

        st.write_stream(stream_words)

        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- [{s['title']}]({s['url']})")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "reasoning": reasoning})
