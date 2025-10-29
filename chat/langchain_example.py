import os
import asyncio
import logging

from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_classic.agents import create_tool_calling_agent, initialize_agent, AgentType, Tool
from langchain_core.tools import tool
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_community.chat_models.mlx import ChatMLX
from langgraph.prebuilt import create_react_agent


def get_current_time() -> str:
    """Gets the current time string"""
    print("TOOL = " + datetime.now().strftime("%H:%M"))
    return datetime.now().strftime("%H:%M")

echo_tool = Tool(
    name="get_current_time",
    func=get_current_time,
    description="Returns the current time in a string",
)

# llm = HuggingFacePipeline.from_model_id(
#     model_id="meta-llama/Llama-3.2-3B-Instruct",
#     task="text-generation",
#     pipeline_kwargs={"max_new_tokens": 20}
# )

logging.basicConfig(level=logging.DEBUG)

llm = MLXPipeline.from_model_id(
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    pipeline_kwargs={"max_tokens": 512, "temp": 0.1},
)

chat_model = ChatMLX(llm=llm)
chat_model = chat_model.bind_tools([get_current_time])
# agent = initialize_agent(llm=chat_model,
#                          agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#                          tools=[echo_tool],
#                          verbose=True)

# # 4. Use an agent that understands tool_calls
# from langchain_core.agents import AgentExecutor, create_tool_calling_agent
# agent = create_tool_calling_agent(model=chat_with_tools, tools=tools)


messages = [
    SystemMessage(
        content="You are a call center assistant Susan which helps patients of St Antonius medical clinic with scheduling appointments."
                "You ask questions one by one to not overload clients with many parallel questions."
                "Based on the question, you might need to make one or more function/tool calls to achieve the purpose."
                "If you decide to invoke any of the function(s), you MUST put it in the JSON format of [{'name':'func_name1','params':{'params_name1':'params_value1', 'params_name2':'params_value2'}}, {'name':'func_name2',...}..]"
                "You SHOULD NOT include any other text in the response."
                "Here is a list of functions in JSON format that you can invoke:"
                "['name': 'get_current_time', 'description': 'Gets the current time', 'params': []]"
    ),
    HumanMessage(content="*phone rings1* Hello?")
]


print(chat_model._to_chat_prompt(messages))

# define your function normally, using `async def` instead of `def`
async def chat():
    while True:
        message = chat_model.invoke(messages)
        text = message.text
        messages.append(SystemMessage(content=text))
        print("AI:", text)
        user_message = input("USER: ")
        if user_message == "bye":
            return
        # now, you can use `await` to call kani's async methods
        messages.append(HumanMessage(content=user_message))


# use `asyncio.run` to call your async function to start the program
asyncio.run(chat())
