import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


os.environ["HUGGINGFACE_API_KEY"] = huggingface_api_key


prompt = ChatPromptTemplate.from_template("""
Use the following piece of context to answer the question asked.
Please try to provide the answer based on the context.

{context}
Question: {input}

Helpful Answer:
""")


groqllm = ChatGroq(
    groq_api_key=groq_api_key,
    model='llama3-8b-8192'
)


hugemb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


loader = CSVLoader("C:/Users/viswe/OneDrive/Desktop/AI-Traveller/updated_file_with_lat_long.csv", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_documents(docs)


db = FAISS.from_documents(documents, hugemb)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question")


conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=groqllm,
    retriever=db.as_retriever(),
    memory=memory
)


user_input = input("Ask your question: ")


response = conversation_chain.invoke({
    'question': user_input,
    'chat_history': memory.buffer 
})


print(response['answer'])
