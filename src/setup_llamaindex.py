
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, SummaryIndex
from llama_index.core import Settings
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI

import os

class LlamaIndexQueryEngine:
    """
    This class is used to setup the LlamaIndex for the application.
    It loads the data from the specified directory and prepares the query engine.
    """
    def __init__(self, data_dir: str, llm_model: str = "gpt-4o-mini"):
        """
        Initialize the LlamaIndexSetup class.
        :param data_dir: The directory where the data is stored.
        """
        # load the data from the specified directory
        self.llm_model = OpenAI(model=llm_model)
        self.root_data_dir = data_dir
        nodes = self.load_documents()
        self.summary_tool = self.prepare_summary_tool(nodes)
        self.vector_tool = self.prepare_vector_tool(nodes)
        self.query_engine = self.prepare_query_engine()


    def load_documents(self):
        documents = []
        records_data_dir = os.path.join(self.root_data_dir, "Record")
        r_documents = SimpleDirectoryReader(records_data_dir).load_data(".pdf")
        textbook_data_dir = os.path.join(self.root_data_dir, "Textbook")
        tb_documents = SimpleDirectoryReader(textbook_data_dir).load_data(".pdf")
        documents = r_documents + tb_documents
        Settings.chunk_size = 1024
        nodes = Settings.node_parser.get_nodes_from_documents(documents)
        return nodes

    def prepare_summary_tool(self, nodes):
        """
        Prepare the query engine for the application.
        """
        summary_index = SummaryIndex(nodes)
        summary_query_engine = summary_index.as_query_engine(
            llm=self.llm_model,
            response_mode="tree_summarize",
            use_async=True,
        )
        return QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            description="Useful for summarization questions related to the data source",
        )

    def prepare_vector_tool(self, nodes):
        vector_index = VectorStoreIndex(nodes)
        vector_query_engine = vector_index.as_query_engine(
            llm=self.llm_model,
        )
        return QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description="Useful for retrieving specific context related to the data source",
        )

    def prepare_query_engine(self):
        return RouterQueryEngine(
            selector=PydanticSingleSelector.from_defaults(),
            query_engine_tools=[
                self.summary_tool,
                self.vector_tool,
            ],
            verbose=True,
        )


