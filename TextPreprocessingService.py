"""Text Preprocessing Service class is responsible for pre-processing pdf files
before persisting them to the DB."""

from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

class TextPreprocessingService:
    def __init__(self, pdf_files, **kwargs):
        """Construct Text Preprocessing Service object.

        Arguments:
            docs: arrat of pdf files.
        """
        self.pdf_files = pdf_files

    def __call__(self, **kwargs):
        """Process pdf files before persisting to DB. This 
        process consists of 2 stages:
            1. Convert pdf files into a Document object
            2. Split Document using RecursiveCharacterTextSplitter

        Returns:
            Array of splitted documents.
        """
        document = self._extract_raw_text()
        return self._split_text(document, **kwargs)

    def _split_text(self, document, **kwargs):
        """Split document using RecursiveCharacterTextSplitter.

        Arguments:
            document: Document object.

        Returns:
            Array of splitted documents.
        """
        text_splitter = RecursiveCharacterTextSplitter(**kwargs)
        return text_splitter.split_documents([document])
    
    def _extract_raw_text(self):
        """Convert pdf files into one Document object.
        Returns:
            Document object.
        """
        raw_text = ""
        for doc in self.pdf_files:
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                raw_text += page.extract_text()
        return Document(raw_text)
    
