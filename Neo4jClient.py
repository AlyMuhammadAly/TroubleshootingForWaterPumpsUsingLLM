from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
# from langchain import Document
URL = "neo4j+s://209b0df6.databases.neo4j.io"
USERNAME = "neo4j"
PASSWORD = "FpzkKnIsv8BecryZZwFd66g2ndcqJWEhdlCVn-x4zkE"

class Neo4jClient():
    def __init__(self, llm) -> None:
        self.db = Neo4jGraph(url=URL, username=USERNAME, password=PASSWORD)
        self.chain = GraphCypherQAChain.from_llm(llm, graph=self.db, verbose=True)

    def to_knowledge_graphs(self, docs):
        # self.db.add_graph_documents(docs)

        self.db.refresh_schema()

    def query_db(self, prompt):
        return self.chain.invoke()