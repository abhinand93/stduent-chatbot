import os
import yaml
import logging
from colorama import init, Fore, Style
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, VECTOR_DB_DIR
from helper import list_to_bullets


# -------------------------------
# Logger setup
# -------------------------------
def setup_logger():
    init(autoreset=True)

    class ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: Fore.CYAN,
            logging.INFO: Fore.YELLOW,
            logging.WARNING: Fore.MAGENTA,
            logging.ERROR: Fore.RED,
            logging.CRITICAL: Fore.RED + Style.BRIGHT,
        }
        def format(self, record):
            color = self.COLORS.get(record.levelno, "")
            return f"{color}{super().format(record)}{Style.RESET_ALL}"

    handler = logging.StreamHandler()
    formatter = ColorFormatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger("chatbot")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


# -------------------------------
# Config loading
# -------------------------------
def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_configs():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    app_config = load_yaml_config(APP_CONFIG_FPATH)
    prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)
    return api_key, app_config, prompt_config


# -------------------------------
# Initialize LLM + Embeddings
# -------------------------------
def init_llm_and_embeddings(app_config, api_key, logger):
    logger.info("Initializing LLM and embeddings...")
    llm = ChatGroq(
        model=app_config["llm"],
        temperature=0.7,
        api_key=api_key,
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=app_config["embedding_model"]
    )
    return llm, embeddings


# -------------------------------
# Initialize Vector Store
# -------------------------------
def init_vectorstore(embeddings, logger):
    logger.info(f"Loading vector store from {VECTOR_DB_DIR}")
    return Chroma(
        collection_name="student_knowledge_base",
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR,
    )


# -------------------------------
# Build System Prompt
# -------------------------------
def build_system_message(ai_tutor_cfg):
    system_message_text = f"""
        {ai_tutor_cfg['role']}

        Main goals:
        - {list_to_bullets(ai_tutor_cfg['goals'])}

        Constraints:
        - {list_to_bullets(ai_tutor_cfg['output_constraints'])}

        Tone & Style:
        - {list_to_bullets(ai_tutor_cfg['style_or_tone'])}
    """
    return SystemMessage(system_message_text)


# -------------------------------
# Retrieval helper
# -------------------------------
def retrieve_relevant_docs(vector_store, query, app_config, logger):
    rag_result = vector_store.similarity_search_with_score(
        query,
        k=app_config["vectordb"]["n_results"]
    )

    threshold = app_config["vectordb"]["threshold"]
    filtered = [(doc, score) for doc, score in rag_result if score >= threshold]

    logger.info(f"Docs in store: {len(vector_store.get()['ids'])}")
    return [doc.page_content for doc, _ in filtered]


# -------------------------------
# Chat loop
# -------------------------------
def run_chatbot():
    logger = setup_logger()
    api_key, app_config, prompt_config = load_configs()

    llm, embeddings = init_llm_and_embeddings(app_config, api_key, logger)
    vector_store = init_vectorstore(embeddings, logger)
    system_message = build_system_message(prompt_config["ai_tutor_system_prompt"])

    MAX_HISTORY_LENGTH = app_config["max_history_length"]
    chat_history = [system_message]

    while True:
        try:
            user_input = input(Fore.CYAN + "You: " + Style.RESET_ALL)
            if user_input.strip().lower() in ["exit", "quit", "q"]:
                logger.info("Exiting chatbot. Goodbye!")
                break

            chat_history.append(HumanMessage(content=user_input))

            retrieved_docs = retrieve_relevant_docs(vector_store, user_input, app_config, logger)
            if retrieved_docs:
                context = "\n\n".join(retrieved_docs)
                chat_history.append(HumanMessage(content=f"Relevant context:\n{context}"))
                logger.info(f"Context retrieved : {context}")
                logger.debug("Context retrieved and added to history.")
            else:
                logger.warning("No relevant context retrieved.")

            response = llm.invoke(chat_history)
            print(Fore.GREEN + "\nChatbot: " + Style.RESET_ALL + response.content)

            chat_history.append(AIMessage(content=response.content))
            chat_history = chat_history[-MAX_HISTORY_LENGTH:]

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")


# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    run_chatbot()
