from mlx_lm import load
from mlx_lm import generate

# Define your model to import
model_name = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"

# Loading model
model, tokenizer = load(model_name,
                        tokenizer_config={"eos_token": "eot_id"} )

# Define the role of the chatbot
chatbot_role = "You are a call center assistant Susan which helps patients of St Antonius medical clinic with scheduling appointments."

# Define a mathematical problem
math_question = "CLIENT: Hello? \nYOU:Good morning! Thank you for calling St. Antonius Medical Clinic. How can I help you? \nCLIENT: I would like to schedule an appointment with doctor Smith. \nYOU:"


# Set up the chat scenario with roles
messages = [
    {"role": "system", "content": chatbot_role},
    {"role": "user", "content": math_question}
]


# Apply the chat template to format the input for the model
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

# Decode the tokenized input back to text format to be used as a prompt for the model
prompt = tokenizer.decode(input_ids)

# Generate a response using the model
response = generate(model, tokenizer, max_tokens=512, prompt=prompt)

# Output the response
print(response)
