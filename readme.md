# ✍️ LangGraph Article Generator

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-green.svg)](https://python.langchain.com/)

An autonomous AI research and writing agent deployed as a clean web application. Powered by **LangGraph**, this agent drafts an article, acts as its own editor to review the content, autonomously searches the live web for missing information, and refines the draft into a final, highly accurate piece.

## ✨ Core Workflow
The application utilizes a cyclic state graph to ensure high-quality outputs:
1. **Drafting Node:** Generates an initial article based on the user's topic.
2. **Review Node:** An AI editor evaluates the draft. It decides if the article is complete or if it lacks specific factual depth.
3. **Search Node (Conditional):** If the editor requests more info, the agent triggers a live web search using the `Tavily API`.
4. **Revision:** The newly acquired web data is fed back into the writing node to improve the draft until the editor is satisfied.

## 🏗️ Architecture Stack
* **LLM Engine:** Mistral AI (`mistral-small-latest`)
* **Agent Framework:** LangChain / LangGraph (`StateGraph`)
* **Search Engine:** Tavily API
* **Web UI:** Streamlit

## 🚀 Quick Start (Local Setup)

### 1. Clone the repository

git clone [https://github.com/akshitgajera1013/LangGraph-Article-Generator.git](https://github.com/akshitgajera1013/LangGraph-Article-Generator.git)

cd LangGraph-Article-Generator

2. Install Dependencies

Ensure you have Python 3.9+ installed, then run:

pip install -r requirements.txt

3. Environment Variables

Create a .env file in the root directory and add your API keys. Never commit this file to GitHub.

MISTRAL_API_KEY="your_mistral_api_key_here"
TAVILY_API_KEY="your_tavily_api_key_here"

4. Run the Application

Start the Streamlit web server:

streamlit run app.py


The UI will automatically open in your default web browser at http://localhost:8501