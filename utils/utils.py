import re
import torch

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def padding(array, max_length, padding_value):
    """
    Input:
        array: list of words in the sequence that need padding
        max_length: output of word list after padding
        padding_value: padding value
    Output:
        a tensor of the sequence after padding
    """
    if len(array) < max_length:
            padding_length = max_length - len(array)
            padding_array = array + [padding_value]*padding_length
            return padding_array
    else:
        return array[:max_length]
    
def collate_fn(batch):
    batch_en_ids = [data["en_ids"] for data in batch]
    batch_de_ids = [data["de_ids"] for data in batch]

    return {
         "en_ids": torch.tensor(batch_en_ids, dtype= torch.long),
         "de_ids": torch.tensor(batch_de_ids, dtype= torch.long)
    }

def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)