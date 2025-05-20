import config
import sentencepiece as spm

def train_tokenizer():
    try:
        spm.SentencePieceTrainer.train(
            input=config.HINMIX_SAMPLE_TXT,
            model_prefix=config.TOKENIZER_MODEL_PREFIX,
            vocab_size=config.GPT_CONFIG["vocab_size"],
            model_type='bpe',
            user_defined_symbols=config.USER_DEFINED_SYMBOLS,
            split_digits=True,
            add_dummy_prefix=False,
            remove_extra_whitespaces=True,
            normalization_rule_name='nfkc',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3
        )
        print(f"Tokenizer model trained and saved with prefix: {config.TOKENIZER_MODEL_PREFIX}")
    except Exception as e:
        print(f"Error during tokenizer training: {e}")

if __name__ == "__main__":
    train_tokenizer()