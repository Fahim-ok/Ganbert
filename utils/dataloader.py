import torch
from torch.utils.data import TensorDataset, DataLoader

def generate_data_loader(input_examples, label_masks, label_map, tokenizer, max_seq_length, batch_size, shuffle=False):
    input_ids, input_masks, label_ids, label_mask_array = [], [], [], []

    for text, label_mask in zip(input_examples, label_masks):
        encoded_sent = tokenizer.encode(text[0], truncation=True, padding="max_length", max_length=max_seq_length)
        input_ids.append(encoded_sent)
        input_masks.append([1] * len(encoded_sent))
        label_ids.append(label_map[text[1]])
        label_mask_array.append(label_mask)

    dataset = TensorDataset(
        torch.tensor(input_ids),
        torch.tensor(input_masks),
        torch.tensor(label_ids, dtype=torch.long),
        torch.tensor(label_mask_array)
    )
    sampler = DataLoader.RandomSampler(dataset) if shuffle else DataLoader.SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)
