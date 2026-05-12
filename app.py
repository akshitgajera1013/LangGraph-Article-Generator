import streamlit as st
import os
import operator
from dotenv import load_dotenv
from typing_extensions import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_mistralai.chat_models import ChatMistralAI
from tavily import TavilyClient
from langchain.tools import tool
from langchain.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

# ---------------- CONFIG & ENV ---------------- #
load_dotenv()
st.set_page_config(page_title="LangGraph Article Writer", page_icon="✍️", layout="centered")

# ---------------- LANGGRAPH DEFINITION ---------------- #
@st.cache_resource
def build_agent():
    # Initialize Model
    model = ChatMistralAI(
        model='mistral-small-latest',
        api_key=os.getenv('MISTRAL_API_KEY')
    )

    # Define Tool
    @tool
    def surfInternet(query: str) -> str:
        """Use this tool to surf the internet for information."""
        client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
        result = client.search(query)
        return str(result)

    tools = [surfInternet]
    model_with_tools = model.bind_tools(tools)

    # Define State
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]

    # Define Nodes
    def write_article_node(state: State):
        """Generates the article based on current context."""
        response = model.invoke(state['messages'])
        return {'messages': [response]}

    def review_article_node(state: State):
        """Reviews the article and decides if a web search is needed."""
        response = model_with_tools.invoke([
            SystemMessage(
                content="You are an expert editor. Review the article you just wrote. "
                        "If more information is needed to improve it, use the surfInternet tool. "
                        "If the article is excellent and needs no more research, provide a final concluding feedback message."
            ),
            *state['messages'],
            HumanMessage(
                content="Review the article you just wrote. If more information is needed, use the surfInternet tool. Otherwise give final feedback."
            )
        ])
        return {'messages': [response]}

    def internet_search_node(state: State):
        """Executes the Tavily search tool."""
        messages = state['messages']
        last_message = messages[-1]
        tool_call = last_message.tool_calls[0]
        query = tool_call["args"]["query"]
        
        res = surfInternet.invoke(query)
        
        return {
            'messages': [
                ToolMessage(
                    content=res,
                    tool_call_id=tool_call["id"]
                )
            ]
        }

    # Define Edge Logic
    def decide_internet_search(state: State):
        last_msg = state['messages'][-1]
        if last_msg.tool_calls:
            return 'internet_search'
        else:
            return END

    # Build and Compile Graph
    agent_builder = StateGraph(State)
    agent_builder.add_node('write_article', write_article_node)
    agent_builder.add_node('review_article', review_article_node)
    agent_builder.add_node('internet_search', internet_search_node)

    agent_builder.add_edge(START, 'write_article')
    agent_builder.add_edge('write_article', 'review_article')
    agent_builder.add_conditional_edges('review_article', decide_internet_search, ['internet_search', END])
    agent_builder.add_edge('internet_search', 'write_article')

    return agent_builder.compile()

# Compile the graph
agent = build_agent()

# ---------------- STREAMLIT UI ---------------- #
st.title("✍️ Autonomous Article Writer")
st.markdown("Powered by **LangGraph**, **Mistral**, and **Tavily**.")

# User Input
topic = st.text_input("Enter your article topic:", placeholder="e.g., The benefits of using AI in education")

if st.button("Generate Article", type="primary"):
    if not topic.strip():
        st.warning("Please enter a topic to begin.")
    else:
        # Use st.status to show a loading container while the graph loops
        with st.status("Agent is researching and drafting...", expanded=True) as status:
            try:
                # 1. Trigger the LangGraph execution
                res = agent.invoke({
                    'messages': [HumanMessage(content=f"Write a comprehensive article about: {topic}")]
                })
                
                status.update(label="Article generation complete!", state="complete", expanded=False)
                
                # 2. Display the output
                st.success("Workflow finished successfully!")
                
                st.markdown("### Final Article Draft & Review")
                
                # Iterate through the state messages to show the workflow process
                # We skip the very first message because it's just the user's prompt
                for msg in res['messages'][1:]:
                    if isinstance(msg, ToolMessage):
                        with st.expander("🔍 Internet Search Results"):
                            st.write(msg.content)
                    elif msg.content:
                        st.markdown(msg.content)
                        st.divider()

            except Exception as e:
                status.update(label="An error occurred", state="error")
                st.error(f"Graph Execution Error: {str(e)}")