from torch.utils.data import Dataset, DataLoader
import torch 

from text_helper import token_to_int_mapping,tokenize_and_pad_sequence

class ReactionDataset(Dataset):
    def __init__(self, dataframe, max_seq_length):
        self.dataframe = dataframe
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        reagent_sequence = tokenize_and_pad_sequence(df['reagent'][idx], max_seq_length=108)
        tensor_reagent = torch.tensor(reagent_sequence).unsqueeze(0)

        solvent_sequence = tokenize_and_pad_sequence(df['solvent'][idx], max_seq_length=108)
        tensor_solvent = torch.tensor(solvent_sequence).unsqueeze(0)

        catalyst_sequence = tokenize_and_pad_sequence(df['catalyst'][idx], max_seq_length=108)
        tensor_catalyst = torch.tensor(catalyst_sequence).unsqueeze(0)


        canonic_rxn = self.dataframe['canonic_rxn'][idx]

        return canonic_rxn,[tensor_reagent,tensor_solvent,tensor_catalyst]
    

reaction_dataset = ReactionDataset(df, max_seq_length=100)
reaction_dataloader = DataLoader(reaction_dataset, batch_size=1, shuffle=True)


