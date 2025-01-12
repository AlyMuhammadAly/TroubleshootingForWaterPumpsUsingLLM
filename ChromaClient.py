"""Chroma Client class is responsible for communicating with ChromaDB."""

from langchain_chroma import Chroma


PERSIST_DIR = "./chroma_db"

class ChromaClient():
    def __init__(self, embedding_function):
        """Construct Chroma Client object and create an instance of 
        ChromaDB on disk.

        Arguments:
            embedding_function: OllamaEmbeddings object.
        """
        self.embedding_function = embedding_function
        self.db = Chroma(persist_directory=PERSIST_DIR, embedding_function=self.embedding_function)

    def convert_to_embeddings_and_save_to_disk(self, docs):
        """Convert splitted documents to embeddings and persist them to
        ChromaDB.

        Arguments:
            docs: array of splitted documents.
        """
        self.db = self.db.from_documents(docs, embedding=self.embedding_function, persist_directory=PERSIST_DIR)
    
    def get_retriever(self):
        """Return DB as retriever to be used by the constructed chain for
        retrieving data from database.

        Returns:
            retriever: retriever object.
        """
        return self.db.as_retriever()
