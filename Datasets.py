import config
from datasets import load_dataset, Dataset as HFDataset

def prepare_hinmix_sample_file():
    print("Preparing HINMIX sample file for tokenizer...")

    hinmix_ds_hicm_train = load_dataset(
        "kartikagg98/HINMIX_hi-en", "lcsalign-hicm", split="train", streaming=True
    )
    hicm_set = list(hinmix_ds_hicm_train.take(config.HINMIX_LCSALIGN_HICM_TAKE))

    hinmix_ds_en_test = load_dataset(
        "kartikagg98/HINMIX_hi-en", "lcsalign-en", split="test", streaming=True
    )
    hinmix_dst_en = list(hinmix_ds_en_test.take(config.HINMIX_LCSALIGN_EN_TAKE))

    hinmix_ds_hi_test = load_dataset(
        "kartikagg98/HINMIX_hi-en", "lcsalign-hi", split="test", streaming=True
    )
    hinmix_dst_hi = list(hinmix_ds_hi_test.take(config.HINMIX_LCSALIGN_HI_TAKE))

    with open(config.HINMIX_SAMPLE_TXT, "w", encoding="utf-8") as f_txt:
        for sample_list in [hicm_set, hinmix_dst_en, hinmix_dst_hi]:
            for sample in sample_list:
                if sample and "text" in sample:
                    f_txt.write(sample["text"] + "\n")
                else:
                    print(f"Warning: Skipping invalid sample: {sample}")
    print(f"HINMIX sample file created at {config.HINMIX_SAMPLE_TXT}")

def create_combined_dataset():
    print("Creating combined dataset...")

    try:
        hindi_discourse_ds_dict = load_dataset(str(config.HINDI_DISCOURSE_SCRIPT_PATH))
        if 'train' in hindi_discourse_ds_dict and isinstance(hindi_discourse_ds_dict['train'], HFDataset):
            hindi_sentences = [
                item['Sentence'] for item in hindi_discourse_ds_dict['train'] if 'Sentence' in item
            ]
        else:
            print("Warning: Could not find 'train' split or 'Sentence' column in Hindi discourse dataset.")
            hindi_sentences = []
    except Exception as e:
        print(f"Error loading Hindi discourse dataset: {e}")
        hindi_sentences = []

    tinystories_train = load_dataset('roneneldan/TinyStories', streaming=True, split='train')
    tinystories_subset_list = list(tinystories_train.take(config.TINYSTORIES_TAKE))
    tinystories_texts = [sample['text'] for sample in tinystories_subset_list if 'text' in sample]

    combined_texts = tinystories_texts + hindi_sentences
    if not combined_texts:
        print("Warning: No texts were combined. Combined dataset will be empty.")
        return

    dataset_hf = HFDataset.from_dict({'text': combined_texts})
    dataset_hf.save_to_disk(str(config.COMBINED_DATASET_PATH))
    print(f"Combined dataset saved to {config.COMBINED_DATASET_PATH}")

if __name__ == "__main__":
    prepare_hinmix_sample_file()
    create_combined_dataset()
    print("Dataset preparation complete.")