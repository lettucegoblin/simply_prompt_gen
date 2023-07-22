from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
import gc

#class named modelManager that will handle all the model loading and generation
class modelManager:
    #constructor
    def __init__(self, modelLocation):
        self.model = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path = modelLocation)
        self.tokenizer = AutoTokenizer.from_pretrained(modelLocation
        )
        # check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # move model to GPU if available
        self.model.to(self.device)
    def generatePrompt(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        attention_mask = input_ids.ne(0).to(self.device)

        # generate prompt
        output = self.model.generate(input_ids, max_length=1000, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id, attention_mask=attention_mask)
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print(output_text)
        return output_text
    #unload model
    def unloadModel(self):
        del self.model, self.tokenizer

        self.model = self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

        
