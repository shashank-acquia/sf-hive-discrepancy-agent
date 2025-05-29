import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGTool:    
    def __init__(self, doc_path: str = None):
        self.doc_path = doc_path or "/Users/shashank.singh/Documents/mycode/sf-hive-discrepancy-agent/resources/rag/BELK-Track-2.docx"
        self.retriever = None
        self.chain = None
        self._initialize()
    
    def _initialize(self):
        try:
            logger.info(f"Loading document from: {self.doc_path}")
            if not os.path.exists(self.doc_path):
                logger.warning(f"Document not found at {self.doc_path}")
                return
            
            if self.doc_path.endswith('.docx'):
                loader = Docx2txtLoader(self.doc_path)
            else:
                loader = TextLoader(self.doc_path)
                
            documents = loader.load()
            
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            
            embedding = OllamaEmbeddings(model="llama3.1:8b")
            vectorstore = FAISS.from_documents(chunks, embedding)
            self.retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
            
            # Create RAG prompt template
            template = """
            You are an expert in SQL, particularly in translating between Hive SQL and Snowflake SQL.
            Use the following pieces of context to help answer the user's question.
            
            Context:
            {context}
            
            Question: {question}
            
            Provide a helpful, detailed response that addresses the question. If the context doesn't contain
            relevant information, say so clearly but still try to provide useful guidance based on your
            general knowledge of SQL and data transformations.
            """
            
            self.prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | llm
                | StrOutputParser()
            )
            
            logger.info("RAG Tool initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG tool: {e}")
            raise
    
    def query(self, question: str) -> str:
        if not self.chain:
            logger.error("RAG Tool not properly initialized")
            return "RAG Tool not properly initialized. Check if document exists."
        
        try:
            logger.info(f"Querying RAG with: {question}")
            result = self.chain.invoke(question)
            return result
        except Exception as e:
            logger.error(f"Error querying RAG: {e}")
            return f"Error querying RAG: {str(e)}"
    
    def get_relevant_context(self, query: str) -> List[Dict]:
        if not self.retriever:
            logger.error("Retriever not initialized")
            return []
        
        try:
            logger.info(f"Finding relevant context for: {query}")
            docs = self.retriever.get_relevant_documents(query)
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    def enhance_sql_suggestion(self, column: str, hive_sql: str, snowflake_sql: str, 
                              hive_val: str, sf_val: str, suggestion: str) -> str:
        if not self.chain:
            return suggestion
        
        query = f"""
        SQL discrepancy in column '{column}'. 
        Hive value: {hive_val}
        Snowflake value: {sf_val}
        Need to fix Snowflake SQL to match Hive output.
        """
        
        try:
            # Get relevant context
            context_docs = self.get_relevant_context(query)
            context = "\n\n".join([doc["content"] for doc in context_docs])
            
            # Create enhanced suggestion prompt
            enhance_prompt = f"""
            I have a SQL discrepancy to fix.
            
            Column: {column}
            Hive value: {hive_val}
            Snowflake value: {sf_val}
            
            Original suggestion:
            {suggestion}
            
            Relevant documentation:
            {context}
            
            Please enhance the original suggestion using the documentation context.
            Focus on improving the SQL fix to better match the documented best practices.
            Make your answer more specific and actionable based on the provided context.
            """
            
            # Get enhanced suggestion
            enhanced_suggestion = self.query(enhance_prompt)
            return enhanced_suggestion
        except Exception as e:
            logger.error(f"Error enhancing suggestion: {e}")
            return suggestion

def create_rag_retriever_tool(rag_tool: RAGTool):
    return create_retriever_tool(
        retriever=rag_tool.retriever,
        name="SQLDiscrepancyDocs",
        description=(
            "Use this tool to search documentation about Hive to Snowflake SQL conversion rules, "
            "common discrepancies, and data type conversions. "
            "Input should be a natural language query about SQL discrepancies."
        )
    )