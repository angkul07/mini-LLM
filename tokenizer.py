import os
import config
import sentencepiece as spm

class Tokenizer:
    def __init__(self, model_path: str = str(config.TOKENIZER_MODEL_FILE)):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Tokenizer model not found at {model_path}. "
            )
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.vocab_size = self.sp.vocab_size()


    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        encoded = self.sp.Encode(text)
        if add_bos:
            encoded = [self.bos_id] + encoded
        if add_eos:
            encoded = encoded + [self.eos_id]
        return encoded

    def decode(self, token_ids: list[int]) -> str:
        return self.sp.Decode(token_ids)
