import asyncio
import torch
import os
from kani import Kani
from kani.engines.huggingface import HuggingEngine

from model.mlx_engine import MlxEngine

engine = HuggingEngine(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    token=os.environ['HUGGINGFACE_AUTH_TOKEN'],
    # suggested args from the Llama model card
    model_load_kwargs={"device_map": "auto", "torch_dtype": torch.bfloat16},
)
# engine = LlamaCppEngine(
#     repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
#     filename="Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf",
#     prompt_pipeline=LLAMA2_PIPELINE
# )

# engine = MlxEngine("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
ai = Kani(engine, system_prompt="You are a call center assistant Susan which helps patients of St Antonius medical clinic with scheduling appointments."
                                "You ask questions one by one to not overload clients with many parallel questions.")

# define your function normally, using `async def` instead of `def`
async def chat_with_kani():
    while True:
        user_message = input("USER: ")
        if user_message=="bye":
            return
        # now, you can use `await` to call kani's async methods
        message = await ai.chat_round_str(user_message)
        if "<|eot_id|>" in message:
            message = message.split("<eoi_id>", 1)[0]
        print("AI:", message)

# use `asyncio.run` to call your async function to start the program
asyncio.run(chat_with_kani())
