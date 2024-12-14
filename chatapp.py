from flask import Flask, request, render_template
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import pickle  # For saving/loading embeddings

# Fetch the GROQ API Key and Ollama API URL from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
ollama_api_url = os.getenv('OLLAMA_API_URL')

# Ensure groq_api_key is available
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# Initialize the ChatGroq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template for the doctor trained on cancer research
prompt = ChatPromptTemplate.from_template(
    """
    You are an AI doctor specialized in cancer diagnosis and treatment, trained on a vast collection of research papers. Your expertise includes lung, brain, chest, colon, and bone marrow cancers. 
    Provide the most accurate and research-based answers to patient questions, keeping in mind their specific conditions.

    <context>
    {context}
    <context>

    Based on the context provided, please offer a detailed and compassionate response to help the patient. Include advice on potential treatment options, diagnostic information, and the importance of follow-up care. 
    Ensure that the answer is aligned with the latest cancer research and medical guidelines.

    Question: {input}
    """
)

# Load the precomputed vectors (embeddings) from a file
def load_precomputed_vectors():
    try:
        with open("vectors.pkl", "rb") as file:
            vectors = pickle.load(file)
        return vectors
    except FileNotFoundError:
        print("Error: vectors.pkl file not found.")
        return None

# Create a global variable for vectors and history
vectors = load_precomputed_vectors()  # Load embeddings when app starts
history = []  # Store user queries and responses

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])  # Fixed route
def index():
    global history
    answer = None
    context = None

    if request.method == 'POST':
        user_prompt = request.form.get('user_prompt')

        if vectors is None:  # Check if vectors are initialized
            answer = "The embedding database is not available."
        else:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            response = retrieval_chain.invoke({'input': user_prompt})
            answer = response['answer']
            context = response.get('context')

            # Add the user prompt and answer to history
            history.append({"prompt": user_prompt, "answer": answer})

    return render_template('index.html', answer=answer, context=context, history=history)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
