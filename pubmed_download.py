from transformers import AutoModel, AutoTokenizer

# Specify the model and tokenizer name
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"

# Load the model and tokenizer
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the path to save the model and tokenizer
save_directory = f"./{model_name}"

# Save the model and tokenizer
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")
