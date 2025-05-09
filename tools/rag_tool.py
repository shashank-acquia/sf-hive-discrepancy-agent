# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.document_loaders import Docx2txtLoader


# loader = Docx2txtLoader("/Users/shashank.singh/Downloads/BELK-Track-2.docx")
# documents = loader.load()


# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# chunks = text_splitter.split_documents(documents)

# # Embed and store in FAISS
# embedding = OllamaEmbeddings(model="llama3.1:8b")
# vectorstore = FAISS.from_documents(chunks, embedding)
# retriever = vectorstore.as_retriever()

# rag_tool = create_retriever_tool(
#     retriever=retriever,
#     name="ReferenceDocs",
#     description="Use this to answer general questions about data validation rules, SQL conversion best practices, and Hive-Snowflake mappings."
# )


