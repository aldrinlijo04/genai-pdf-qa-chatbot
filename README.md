## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:

In many cases, users need specific information from large documents without manually searching through them. A question-answering chatbot can address this problem by:

1. Parsing and indexing the content of a PDF document.
2. Allowing users to ask questions in natural language.
3. Providing concise and accurate answers based on the content of the document.
  
The implementation will evaluate the chatbot’s ability to handle diverse queries and deliver accurate responses.

### DESIGN STEPS:

#### STEP 1: Load and Parse PDF
Use LangChain's DocumentLoader to extract text from a PDF document.

#### STEP 2: Create a Vector Store
Convert the text into vector embeddings using a language model, enabling semantic search.

#### STEP 3: Initialize the LangChain QA Pipeline
Use LangChain's RetrievalQA to connect the vector store with a language model for answering questions.

#### STEP 4: Handle User Queries
Process user queries, retrieve relevant document sections, and generate responses.

#### STEP 5: Evaluate Effectiveness
Test the chatbot with a variety of queries to assess accuracy and reliability.


### PROGRAM:
```
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

def load_db(file, chain_type, k, llm_name):
    """
    Function to load postpartum-specific documents and create a conversational retrieval chain.

    Args:
        file (str): Path to the postpartum-related PDF.
        chain_type (str): Type of retrieval chain.
        k (int): Number of retrieved documents.
        llm_name (str): OpenAI model name (e.g., "gpt-4").
    
    Returns:
        qa: Conversational retrieval chain.
    """
    # Step 1: Load postpartum documents
    loader = PyPDFLoader(file)
    documents = loader.load()

    # Step 2: Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # Step 3: Define embeddings for postpartum content
    embeddings = OpenAIEmbeddings()

    # Step 4: Create vector database for document search
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)

    # Step 5: Configure the retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Step 6: Create a conversational retrieval chain for postpartum topics
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0.5),  # Empathetic response style
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa

```

### OUTPUT:

![image](https://github.com/user-attachments/assets/78f8922a-3146-403f-9bb5-bacb1b753f66)


### RESULT:
Thus, a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain was implemented and evaluated for its effectiveness by testing its responses to diverse queries derived from the document's content successfully.
