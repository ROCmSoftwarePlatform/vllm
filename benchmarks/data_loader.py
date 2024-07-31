from datasets import load_dataset
from itertools import chain

def get_input_sentences(batch_size, block_size, dataset_name, dataset_config_name, tokenizer):
    raw_datasets = load_dataset(dataset_name, dataset_config_name)
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        tokens=[]
        for i in range(len(examples[text_column_name])):
            token = tokenizer.encode(examples[text_column_name][i])
            tokens.append(token)
        return {"input_ids" : tokens}

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=128,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=128,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    input_samples = []
    for i, sample in enumerate(lm_datasets["train"]):
        if i == batch_size:
            break
        input_samples.append(sample["input_ids"])
    return input_samples
