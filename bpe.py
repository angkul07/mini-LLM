# from tokenizers import Tokenizer
import sentencepiece as spm

# class HindiTokenizer:
#     def __init__(self, model_path="/home/angkul/my_data/coding/agi/hindi_GPT/hindi_tokenizer.json"):
#         self.tokenizer = Tokenizer.from_file(model_path)

#     def encode(self, text):
#         return self.tokenizer.encode(text).ids

#     def decode(self, token_ids):
#         return self.tokenizer.decode(token_ids)

MODEL = "/home/angkul/my_data/coding/agi/hindi_GPT/hinglish_32k.model"
sp = spm.SentencePieceProcessor()
sp.load(MODEL)

class HindiTokenizer:
    def encode(self, text):
        return sp.Encode(text)
    
    def decode(self, token_ids):
        return sp.Decode(token_ids)
        