from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

MAX_HISTORY_LENGTH = 10

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')


persistent_directory = "student_knowledge_base/vectorstore"

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0.7,
    api_key=api_key,
)


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


vector_store = Chroma(
    collection_name="student_knowledge_base",
    embedding_function=embeddings,
    persist_directory=persistent_directory
)

system_message = SystemMessage(
                    """You are a friendly and knowledgeable AI tutor designed to help students resolve their doubts across academic subjects.  
                       Your main goals are:
                        1. **Clarity & Simplicity** – Explain concepts in a clear, step-by-step manner, avoiding unnecessary jargon.  
                        2. **Guidance, not just answers** – Encourage students to think critically by guiding them toward the solution instead of directly giving final answers (unless explicitly requested).  
                        3. **Adaptability** – Adjust your explanations to the student’s level (beginner, intermediate, or advanced) and provide real-life examples wherever possible.  
                        4. **Encouragement** – Motivate students, acknowledge their effort, and give hints or learning tips to boost confidence.  
                        5. **Multimodal Support** – When possible, use tables, bullet points, diagrams (in text), or step breakdowns for better understanding.  
                        6. **Boundaries** – Never provide harmful, misleading, or inappropriate content. If a question is outside academics, politely guide the student back to learning.  

                        Tone: Polite, supportive, approachable (like a patient mentor).  
                        Output Style: Clear explanations, step-by-step reasoning, with optional examples or practice exercises.  

                        Always ask clarifying questions if the student’s doubt is vague, and encourage them to try before showing the final solution.
                    """
                )

chat_history = [system_message]

while True:
    try:
        user_input = input("You : ")
        if user_input.strip().lower() in ["exit", "quit", "q"]:
            print("Bye! Have a great day!")
            break

        chat_history.append(HumanMessage(content=user_input))

        rag_result = vector_store.similarity_search_with_score(
            user_input,
            k=3
        )

        print("\n\nTotal docs in store:", len(vector_store.get()['ids']))

        retrieved_docs = [doc.page_content for doc, _ in rag_result]
        
        context = "\n\n".join(retrieved_docs)

        print("#"*100)
        print(f"\n Context : {context}")
        print("#"*100)

        if context:
            chat_history.append(HumanMessage(content=f'Relevant context : \n {context}'))


        response = llm.invoke(chat_history)

        print("#"*100)
        print(f"\n\n Chatbot : {response.content}")

        chat_history.append(AIMessage(content=response.content))
        chat_history = chat_history[-MAX_HISTORY_LENGTH:]
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
