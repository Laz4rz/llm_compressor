# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

text = "Qwen is a"
input_ids = tokenizer.encode(text, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=1)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)
