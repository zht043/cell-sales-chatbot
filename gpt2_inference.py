def generate_n_text_samples(model, tokenizer, input_text):
    text_ids = tokenizer(input_text, return_tensors = 'pt')
    text_ids = text_ids
    model = model

    generated_text_samples = model.generate(
        **text_ids, 
        max_length= 50,  
        num_return_sequences= 1,
        no_repeat_ngram_size= 2,
        repetition_penalty= 1.5,
        top_p= 0.92,
        temperature= .85,
        do_sample= True,
        top_k= 125,
        early_stopping= True
    )
    for t in generated_text_samples:
        text = tokenizer.decode(t, skip_special_tokens=True)
        return text

    
from transformers import AutoConfig
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model_articles_path = './GPT2'
new_model = GPT2LMHeadModel.from_pretrained(model_articles_path)
new_tokenizer = GPT2Tokenizer.from_pretrained(model_articles_path)
bos = new_tokenizer.bos_token
eos = new_tokenizer.eos_token
sep = new_tokenizer.sep_token
print("Chatbot: Hello, how can I help you?")
q = input("User: ")
while q != "exit":
    q_new = ' '.join([bos, q,  sep])
    content = generate_n_text_samples(new_model, new_tokenizer, q_new)
    content = content[len(q)+1:]
    print(f"Chatbot:{content}")
    print("")
    q = input("User: ")