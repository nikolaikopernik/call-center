from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_community.chat_models.mlx import ChatMLX
from langchain_classic.agents import initialize_agent, Tool
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.schema import HumanMessage, SystemMessage, AIMessage

# --- 1. Initialize the MLX model / pipeline ---
# Use the model ID you have (for example, a quantized llama3 model in MLX)
# e.g. "your-llama3-quantized-model" or a HuggingFace / MLX community ID
llm = MLXPipeline.from_model_id(
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    pipeline_kwargs={"max_tokens": 128, "temperature": 0.0,
                     "stop": ["\nFinal Answer:", "\nObservation:"]},
)

# Wrap it to a chat model interface
chat = ChatMLX(llm=llm)

# --- 2. Define a simple tool (for demo) ---
def echo_tool_func(text: str) -> str:
    return f"Echo1234: {text}"

echo_tool = Tool(
    name="echo",
    func=echo_tool_func,
    description="Returns the same text prefixed with 'Echo:'",
)

# --- 3. Initialize agent with that tool ---
agent = initialize_agent(
    tools=[echo_tool],
    llm=chat,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- 4. Use the agent with a prompt that triggers tool usage ---
resp = agent.invoke("Use the echo tool to echo 'Hello from agent!'")
print("Agent response:", resp)
