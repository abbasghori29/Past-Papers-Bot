from flask import Flask, jsonify,redirect,url_for
import os
import requests


import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
import os


os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ["GOOGLE_API_KEY"] = "AIzaSyAlcTTcIez3AHwN0RifX6lPgs2OrKEMNUQ"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=100,
)
google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
new_db = FAISS.load_local("faiss_past_papers/", google_embeddings,allow_dangerous_deserialization=True)

@app.route('/quick_replies', methods=['POST'])
def quick_replies():
    data = request.get_json()
    session_history = data.get('session_history')

    if not session_history or not isinstance(session_history, list):
        return jsonify({"error": "Invalid session history provided"}), 400

    # Generate quick replies based on session history
    session_context = "\n".join(session_history)  # Join all past conversation history

    # Craft a prompt to generate quick replies in a comma-separated format
    prompt_template = """
    Based on the following conversation history, suggest a list of quick replies (questions or actions) that would help continue the conversation effectively. 
    Provide the quick replies as a comma-separated list with no additional explanation.

    Conversation History:
    {session_history}

    Response:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["session_history"])
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    quick_replies = chain.invoke({"session_history": session_context})
    
    # Split the comma-separated quick replies and transform them into a list
    quick_replies_list = [reply.strip() for reply in quick_replies.split(',') if reply.strip()]

    return jsonify({"quick_replies": quick_replies_list})

def getSimilar_documents(query):
    relevant_docs = new_db.similarity_search(query,k=20)
    
    # Convert documents into the desired format
    documents = []
    for doc in relevant_docs:
        meta = doc.metadata.get('file_path', '').replace('\\', '/')  # Extract metadata
        content_preview = doc.page_content[:200]  # Extract first 200 characters of content
        document = {
            "metadata": meta,
            "content_preview": content_preview
        }
        documents.append(document)
    
    return documents


def generate_response(query):
    context=getSimilar_documents(query)
    context=json.dumps(context, indent=2)
    prompt_template = """
You are an intelligent assistant designed to assist users in finding past paper links based on their queries. Your task is to:

- Analyze the provided document objects containing all data (metadata and content) for each paper.

documents: {context}

- Use the metadata and the first 200 characters of the content to understand the document’s subject, type, and relevant details.
- Analyze the user query to understand the specific past paper being requested (e.g., subject, year, type).
- Compare the user query against the metadata and content previews within the document objects provided.
- Identify the document that best matches the query.
- Provide the document link and a brief explanation of the document’s relevance based on its metadata and content.

user question:{question}

**Important Instructions**:
1. Prepend all document links with `C:/Users/Abbas/Desktop/pp project test/` before including them in the output.
2. Only include document links and summaries explicitly mentioned in the provided metadata and content.
3. Do not extrapolate or add speculative information, such as "[And so on]."
4. When a user greets (e.g., "hello," "hi"), respond warmly and politely, such as:  
   - "Hello! How can I assist you today?"  
   - "Hi there! Let me know what you're looking for, and I'll help you find it."
5. Ensure the output format follows the example provided below exactly.

**Output Format**:

Here's everything related to [user's query or topic]:

Below are the relevant documents, along with their links and a brief description:

 <div class="container mt-5">
   <div 
    class="card mb-3 w-100" 
    style="border: 2px solid; 
           border-image-slice: 1; 
           border-width: 2px; 
           border-image-source: linear-gradient(135deg, #6a0dad, #ff00ff); 
           border-radius: 50px;"
   >
    <div class="d-flex align-items-center p-4">
     <img alt="PDF icon with text 'PDF' in the center" class="rounded-circle" height="50" src="https://storage.googleapis.com/a1aa/image/IcdeS20aCywfEUzkV2OHvrSYYhR32htvrbDDjfvCaf2q6fBfE.jpg" width="50"/>
     <div class="ml-4">
      <b class="h5 text-dark">
       Document Title 1 (make some title based on pdf name)
      </b>
      <a class="d-block mt-2 text-danger" href="C:/Users/Abbas/Desktop/pp project test/[here you put relative path given in meta data]" target="_blank">
       <i class="fas fa-file-pdf pr-2">
       </i>
       Link to Document 1
      </a>
      <p class="text-muted mt-2">
       <i>
        Overview:
       </i>
       [Provide document content summary].
      </p>
     </div>
    </div>
   </div>
  </div>

[Repeat as necessary for other documents.]

If no match is found:

"I couldn't find a document matching your query. Please provide more details, such as the subject, year, or type of paper you're looking for."
"""




    PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
    output_parser = StrOutputParser()

    chain = PROMPT | llm | output_parser

    response = chain.invoke({"context": context, "question": query})
    print(response)
    formatted_response = format_response(response)

    return response

def format_response(response):
    # Start with a clean header
    formatted_response = "Here are the results for your query:\n\n"

    # Check if the response contains valid documents
    if "I couldn't find a document matching your query" in response:
        formatted_response += f"Sorry, no relevant documents were found based on your query.\n"
    else:
        formatted_response += "Below are the relevant documents with brief descriptions:\n\n"
        
        # You can further format the response by splitting it into parts and adding line breaks
        docs = response.split("\n\n")  # Assuming each document is separated by a double line break

        for idx, doc in enumerate(docs, 1):
            formatted_response += f"Document {idx}:\n{doc}\n\n"

    return formatted_response
@app.route('/query_paper', methods=['POST'])
def query_paper():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    response = generate_response(query)
    return jsonify({"response": response})

@app.route('/')  
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) 
