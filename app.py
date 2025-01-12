"""This is the main application file."""

import streamlit as st
from TextPreprocessingService import TextPreprocessingService
from langchain_community.embeddings import OllamaEmbeddings
from ChromaClient import ChromaClient
from ConversationService import ConversationService
from langchain.llms import Ollama
import torch
from transformers import AutoTokenizer
from lxt.models.llama import LlamaForCausalLM, attnlrp
from lxt.utils import pdf_heatmap, clean_tokens
from langchain_core.messages import HumanMessage
import langchain

# MODEL = "llama3.1:latest"
MODEL = "llama3.2:1b"
EMBEDDINGS_MODEL = "nomic-embed-text:v1.5"
# Temperature for Llama-3.1
TEMP = 0

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBEDDINGS = "Embeddings"

# Ollama URL
BASE_URL = "http://localhost:11434"

def init():
	"""Initialise conversation history, large langugage model object, ChromaDB client and
	embedding function. All these were stores in the session_state object to avoid
	losing the state of the variables.
	"""
	# Uncomment to debug.
	# langchain.debug = True

	embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
	st.session_state.chroma_client = ChromaClient(embeddings)

	st.session_state.llm = Ollama(base_url=BASE_URL, model=MODEL, temperature=TEMP)

	st.session_state.history = []

def main():
	"""Main method that is responsible for running the streamlit app."""
	admin_view_tab, customer_view_tab = st.tabs(["Admin View", "Customer View"])

	with admin_view_tab:
		st.header("Upload PDFs to Knowledge Base")
		pdfs = st.file_uploader("Upload your files here", accept_multiple_files=True)

		# TODO: Add support to KG
		save_text_as = st.selectbox("How would you like your PDFs to be saved as?", ("Embeddings", "Knowledge Graph"),
							   		index=0, placeholder="Select contact method...",)
		if st.button("Upload"):
			with st.spinner("Preprocessing....."):
				init()

				tps = TextPreprocessingService(pdfs)
				splitted_docs = tps(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
				
				if save_text_as == EMBEDDINGS:
					st.session_state.chroma_client.convert_to_embeddings_and_save_to_disk(splitted_docs)
					st.session_state.retriever = st.session_state.chroma_client.get_retriever()

	with customer_view_tab:
		st.header("Troubleshooting Agent")
		prompt = st.chat_input("Enter your text")

		if prompt:
			conversation_service = ConversationService(st.session_state.llm, st.session_state.retriever)
			st.session_state.chain = conversation_service.create_chain()

			answer = st.session_state.chain.invoke({"question": prompt, "history": st.session_state.history})
			st.session_state.history.append([HumanMessage(content=prompt), answer])

			st.write(f"User has sent the following prompt: {prompt}")
			st.write(f"LLM responds with the following statement: {answer}")

if __name__ == '__main__':
	main()