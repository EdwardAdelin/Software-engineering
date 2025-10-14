from transformers import AutoTokenizer, AutoModelForCausalLM
model_name =  "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#prepare input
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
#generate output
#outputs = model.generate(**inputs, max_new_tokens=50)

#add some extra arguents temp, top k, top_p, do_sample, repetition_penalty etc.
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.95, top_k=50, top_p=0.95, do_sample=True, repetition_penalty=1.2)


#decode and print the output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))