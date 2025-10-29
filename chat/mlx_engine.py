from typing import AsyncIterable

from kani import ChatMessage, AIFunction
from kani.engines import BaseEngine
from kani.engines.base import BaseCompletion, Completion
from mlx_lm import load, generate, stream_generate
import time

class MlxEngine(BaseEngine):
    def __init__(
            self,
            model_id: str):
        self.model, self.tokenizer = load(model_id, tokenizer_config={"eos_token": "eot_id"} )
        self.max_context_size = 1024 * 8

    async def predict(self, messages: list[ChatMessage], functions: list[AIFunction] | None = None,
                      **hyperparams) -> BaseCompletion:
        messages = [ {"role": m.role.name, "content": m.content} for m in messages ]
        # Apply the chat template to format the input for the model
        input_ids = self.tokenizer.apply_chat_template(messages)

        # Decode the tokenized input back to text format to be used as a prompt for the model
        prompt = self.tokenizer.decode(input_ids)

        # Generate a response using the model
        start = time.time()
        response = generate(self.model, self.tokenizer, max_tokens=512, prompt=prompt)
        print(time.time()-start)
        if "<|eot_id|>" in response:
            response = response.split("<|eot_id|>")[0]
        return Completion(ChatMessage.system(response))

    async def stream(self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams) -> AsyncIterable[str | BaseCompletion]:
        messages = [ {"role": m.role.name, "content": m.content} for m in messages ]
        # Apply the chat template to format the input for the model
        input_ids = self.tokenizer.apply_chat_template(messages)

        # Decode the tokenized input back to text format to be used as a prompt for the model
        prompt = self.tokenizer.decode(input_ids)

        # Generate a response using the model
        start = time.time()

        stream_generate(self.model, self.tokenizer, max_tokens=200, prompt=prompt)

    def message_len(self, message: ChatMessage) -> int:
        return len(self.tokenizer.encode(message.text))
