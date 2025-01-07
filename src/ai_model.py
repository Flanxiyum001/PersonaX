from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to generate text based on a prompt
def generate_text(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":  
    prompt = "Hello, I am Flanx and I love coding. My favorite hobby is"
    response = generate_text(prompt)
    print(response)
    