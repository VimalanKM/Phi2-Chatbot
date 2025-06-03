from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2')

chat = pipeline('text-generation', model=model, tokenizer=tokenizer)

while True:
    user_input = input('You: ')
    if user_input.lower() in ['exit', 'quit']:
        break
    response = chat(user_input, max_new_tokens=100, do_sample=True)[0]['generated_text']
    print('Bot:', response[len(user_input):].strip())