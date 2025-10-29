"""
Manual minimal agent loop for MLX + LLaMA-3.2 (ChatMLX)
- Parses tool calls produced by the model and executes them directly.
- Stops when the model emits "Final Answer: <...>"
"""
import asyncio
import re
import json
import sys

from datetime import datetime
from typing import Callable, Dict, Any, Tuple, Optional
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_community.chat_models.mlx import ChatMLX
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage, AIMessage



# ---- Replace these imports with your actual MLX + ChatMLX classes ----
# from langchain_community.llms.mlx_pipeline import MLXPipeline
# from langchain_community.chat_models.mlx import ChatMLX
#
# Example:
# llm = MLXPipeline.from_model_id("your-llama3-2-id", pipeline_kwargs={"max_tokens":512})
# chat = ChatMLX(llm=llm)
# ----------------------------------------------------------------------

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# ---------------- Example tool implementations ----------------
def tool_echo(text: str) -> str:
    return f"Echo1234: {text}"

def tool_add(a: float, b: float) -> str:
    # simple calculator tool
    return str(a + b)

def get_current_time() -> str:
    # simple calculator tool
    print("TOOOL")
    return "Current time is " + datetime.now().strftime("%H:%M")

def tool_search(query: str) -> str:
    # placeholder - replace with real web/search function
    return f"[search results for '{query}' (simulated)]"

# map tool names to functions and a small schema for parsing arguments
TOOLS: Dict[str, Tuple[Callable[..., str], Dict[str, type]]] = {
    "echo": (tool_echo, {"text": str}),
    "add": (tool_add, {"a": float, "b": float}),
    "search": (tool_search, {"query": str}),
    "get_current_time": (get_current_time, {}),
}

def try_parse_json_block(text: str) -> Optional[list[Dict[str, Any]]]:
    """Look for a ```json {...}``` block and parse it into dict."""
    try:
        return json.loads(text.replace("'","\""))
    except json.JSONDecodeError as e:
        eprint("Invalid JSON syntax:", e)
        return None

# ---------------- The manual agent loop ----------------
class ManualAgent:
    def __init__(self, chat_model):
        """
        chat_model: an object with a .generate or .call method that accepts messages.
                    Expected to accept a "messages" style list of dicts:
                    [{"role":"system","content":...},{"role":"user","content":...}, ...]
                    Adjust send_prompt() to match your model wrapper.
        """
        self.chat = chat_model
        self.system_prompt = """
            You are a call center assistant Susan which helps patients of St Antonius medical clinic with scheduling appointments.
            You ask questions one by one to not overload clients with many parallel questions.
            You don't know the current time until you call a special tool for that.
        """
        self.function_question_prompt = """
            Based on the user input above, check if you need to use one of the following tools:
            Here is a list of functions in JSON format that you can invoke:
             - If you need to know the time now (current time), do not rely on your training information and use '[{'name':'get_current_time','params':{}}]'
             - If you need to sum two numbers a and b then use '[{'name':'add','params':{'a':number,'b':number}}]'
            In the response, print only a valid JSON with the list of tools in the format above: '[{...},{...}]'. You SHOULD NOT use any other text in the response except of JSON. If you don't need any tool data, just print 'NO'.
        """
        # conversation memory
        self.messages = [
            SystemMessage(self.system_prompt),
        ]

    def run_tool(self, tool_name: str, args: dict)->str:
        eprint(f"[Agent] Parsed action: {tool_name} with args: {args!r}")
        if tool_name not in TOOLS:
            return f"Error: unknown tool '{tool_name}'"
        else:
            func, schema = TOOLS[tool_name]
            # Normalize args:
            args_dict = args

            # Cast args to expected types (best-effort)
            casted = {}
            for k, t in schema.items():
                if k in args_dict:
                    try:
                        casted[k] = t(args_dict[k])
                    except Exception:
                        # fallback: pass original
                        casted[k] = args_dict[k]
            try:
                return func(**casted)
            except TypeError as e:
                return f"Failed to call tool {tool_name}: " + e

    def send_prompt(self, messages) -> BaseMessage:
        return self.chat.invoke(messages)

    def _parse_model_output(self, message: BaseMessage) -> BaseMessage:
        # Check final answer first (so if model finishes, we stop)
        text = message.text
        if not text.startswith("[") and not text.startswith("{"):
            return message

        # JSON block parsing (preferred)
        parsed_json = try_parse_json_block(text)
        message = "Use the results of this tools to answer the following user message:"
        if parsed_json:
            if not isinstance(parsed_json, list):
                parsed_json=[parsed_json]
            for item in parsed_json:
                # expected shape: {"name":"tool_name","params":{...}}
                tool_name = item["name"]
                args = item["params"]
                message += f"Tool {tool_name}, response: {self.run_tool(tool_name, args)}"

        return SystemMessage(content=message)

    def run(self, user_query: str, max_steps: int = 3)->BaseMessage:
        func_asked = HumanMessage(user_query + "\n\n" + self.function_question_prompt)
        eprint(f"[FUNC] - {func_asked}")
        func_chat = [self.system_prompt, func_asked]

        raw_reply = self.send_prompt(func_chat)  # model text output
        eprint(f"[Model reply]:{raw_reply.text}")

        parsed = self._parse_model_output(raw_reply)

        if isinstance(parsed, SystemMessage):
            eprint(f"[AI] - {parsed.text}")
            self.messages.append(parsed)

        # parsed_type == "none" -> model didn't produce parseable action or final answer
        # strategy: ask the model to reformat (simple fix)
        eprint(f"[USER:] - {user_query}")
        self.messages.append(HumanMessage(user_query))
        return self.send_prompt(self.messages)





# --- 1. Initialize the MLX model / pipeline ---
# Use the model ID you have (for example, a quantized llama3 model in MLX)
# e.g. "your-llama3-quantized-model" or a HuggingFace / MLX community ID
llm = MLXPipeline.from_model_id(
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    pipeline_kwargs={"max_tokens": 128, "temperature": 0.3,
                     "stop": ["\nFinal Answer:", "\nObservation:"]},
)

# Wrap it to a chat model interface
chat = ChatMLX(llm=llm)
# adapt to your chat model wrapper; we implement send_prompt by delegation
agent = ManualAgent(chat_model=chat)


# define your function normally, using `async def` instead of `def`
async def chat34():
    message = agent.run("Hello?")
    while True:
        text = message.text
        print("AI:", text)
        user_message = input("USER: ")
        if user_message == "bye":
            return
        # now, you can use `await` to call kani's async methods
        message = agent.run(user_message)

if __name__ == "__main__":
    asyncio.run(chat34())
