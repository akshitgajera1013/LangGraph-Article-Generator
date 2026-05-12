from langgraph.graph import StateGraph,START,END
from langchain_mistralai.chat_models import ChatMistralAI
from tavily import TavilyClient
from langchain.tools import tool
import os
from dotenv import load_dotenv
load_dotenv()
import operator
from typing_extensions import Annotated,TypedDict
from langchain.messages import AnyMessage,SystemMessage,HumanMessage,ToolMessage



model=ChatMistralAI(
    model='mistral-small-latest',
    api_key=os.getenv('MISTRAL_API_KEY')
    )


@tool
def surfInternet(query:str)->str:
    """Use this tool to surf the internet for information. The query is a string that describes what you want to search for. Return the search results as a string.
    """

    client=TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
    result=client.search(query)
    return str(result)

tools=[surfInternet]
tools_dict={tool.name:tool for tool in tools}

model_with_tools=model.bind_tools(tools)

class State(TypedDict):
    messages:Annotated[list[AnyMessage], operator.add]


def write_article_node(state:State):
    """Write an article about the benefits of using AI in education. Use the surfInternet tool to gather information about the topic. Return the article as a string.
    """

    response=model.invoke(state['messages'])

    return {
        'messages':[response]}


def review_article_node(state: State):

    response = model_with_tools.invoke([
        
        SystemMessage(
            content="You are a helpful assistant that reviews articles about AI in education."
        ),

        *state['messages'],

        HumanMessage(
            content="""
Review the article you just wrote.

If more information is needed,
use the surfInternet tool.

Otherwise give final feedback.
"""
        )
    ])

    return {
        'messages': [response]
    }


def internet_search_node(state:State):
    """Use the surfInternet tool to gather more information about the benefits of using AI in education. Return the search results as a string.
    """

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


def decide_internet_search(state:State):
    last_msg=state['messages'][-1]

    if last_msg.tool_calls:
        return 'internet_search'
    else:
        return END
        


agent_builder=StateGraph(State)

agent_builder.add_node('write_article',write_article_node)
agent_builder.add_node('review_article',review_article_node)
agent_builder.add_node('internet_search',internet_search_node)


agent_builder.add_edge(START,'write_article')
agent_builder.add_edge('write_article','review_article')
agent_builder.add_conditional_edges('review_article',decide_internet_search,['internet_search',END])
agent_builder.add_edge('internet_search','write_article')

agent=agent_builder.compile()

res=agent.invoke({
    'messages':[HumanMessage(content="Write an article about the benefits of using AI in education.")]
})

print(res['messages'][-1].pretty_print())