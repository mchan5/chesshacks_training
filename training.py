from transformers inport AutoModel 

model.save_pretrained("./mychess-model") 
model.push_to_hub("your-username/chess-bot-model") 

sr