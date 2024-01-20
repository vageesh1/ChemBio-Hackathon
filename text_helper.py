import torch

def pad_sequence(sequence, max_length, padding_token='<PAD>'):
    return sequence + [padding_token] * (max_length - len(sequence))

token_to_int_mapping = {}

def token_to_int(token):
    global token_to_int_mapping

    if token not in token_to_int_mapping:
        # Assign a random number to the new token
        random_int = len(token_to_int_mapping) + 1  # Start from 1 to avoid conflict with 0 for padding
        token_to_int_mapping[token] = random_int

    return token_to_int_mapping[token]

def tokenize_and_pad_sequence(sequence, max_seq_length):
    # Convert <PAD> values to 0
    tokenized_sequence = [0 if token == '<PAD>' else token_to_int(token) for token in sequence]

    # Pad the sequence with zeros if it's shorter than max_seq_length
    padded_sequence = tokenized_sequence + [0] * (max_seq_length - len(tokenized_sequence))

    # Convert the sequence to a PyTorch tensor
    tensor_sequence = torch.tensor(padded_sequence, dtype=torch.long)

    return tensor_sequence

