"""Conversation Service class is responsible for constructing conversation
chain for language model."""

from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import langchain
from langchain_core.prompts import PromptTemplate

class ConversationService():
    def __init__(self, llm, retriever) -> None:
        """Construct Conversation Service.

        Arguments:
            llm: Ollama object.
            retriever: retriever object.
        """
        self.llm = llm
        self.retriever = retriever

    def create_chain(self):
        """Construct chain.

        Returns:
            chain: chain of all processes.
        """
        chain = (
            {
                "context": self.retriever,
                "question": itemgetter("question"),
                "history": itemgetter("history")
            }
            | self._get_prompt()
            | self.llm
            | self._get_parser()
        )
        return chain
    
    def _get_parser(self):
        """Construct a StrOutputParser object.

        Returns:
            StrOutputParser object.
        """
        return StrOutputParser()
    
    def _get_prompt(self):
        """Creates prompt template for a language model following CO-STAR framework.
        CO-STAR stands for: Context, Objective, Style, Audience and Response.

        Returns:
            PromptTemplate object.
        """
        template = """
            # CONTEXT #
            {context}

            #############

            # OBJECTIVE #
            Act as a professional water pump technician from Stuart Turner company.
            Your job is to help me troubleshoot my pumps and answer all my queries.
            Answer based on the given context and chat history. Don't reply for out of context questions.

            History: {history}
            Question: {question}

            #############

            # STYLE #

            Follow the simple writing style common in communications aimed at consumers

            #############

            # AUDIENCE #
            Tailor your response towards consuemrs who are looking for troublshooting
            their water pumping system.

            #############

            # RESPONSE #
            Be concise and succinct in your response yet impactful. Where appropriate, use
            appropriate emojies.
        """
        return PromptTemplate.from_template(template)
