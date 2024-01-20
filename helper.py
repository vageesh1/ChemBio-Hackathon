from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import spacy
import fasttext

import torch
import torch.nn as nn
import torchtext
import torch.nn.functional as F
from torch.utils.data import Dataset,dataloader
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader

import numpy as np
import pandas as pd

def featurize_molecule(mol):
    # Compute Morgan fingerprints for each atom
    atom_features = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom_feature = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, atomIndices=[idx])
        atom_features.append(np.array(atom_feature))

    return np.array(atom_features)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # Add explicit hydrogens
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates for visualization
    AllChem.EmbedMolecule(mol, randomSeed=42)  # You can choose any seed value

    # Get atom features and adjacency matrix
    num_atoms = mol.GetNumAtoms()
    atom_features = np.zeros((num_atoms, 3))  # You may need to adjust the feature dimensions
    adjacency_matrix = np.zeros((num_atoms, num_atoms))

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1  # Adjacency matrix is symmetric

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom_features[idx, 0] = atom.GetAtomicNum()  # Atom type or atomic number
        atom_features[idx, 1] = atom.GetTotalNumHs()  # Number of hydrogen atoms
        atom_features[idx, 2] = atom.GetFormalCharge()  # Formal charge

    # Convert to PyTorch tensors
    atom_features = torch.tensor(atom_features, dtype=torch.float)

    # Create edge_index using the adjacency matrix
    edge_index = torch.tensor(np.column_stack(np.where(adjacency_matrix)), dtype=torch.long)

    # Create PyTorch Geometric data object
    data = Data(x=atom_features, edge_index=edge_index.t().contiguous())  # Transpose edge_index

    return data

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # Add explicit hydrogens
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates for visualization
    AllChem.EmbedMolecule(mol, randomSeed=42)  # You can choose any seed value

    # Get atom features and adjacency matrix
    num_atoms = mol.GetNumAtoms()
    atom_features = np.zeros((num_atoms, 3))  # You may need to adjust the feature dimensions
    adjacency_matrix = np.zeros((num_atoms, num_atoms))

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1  # Adjacency matrix is symmetric

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom_features[idx, 0] = atom.GetAtomicNum()  # Atom type or atomic number
        atom_features[idx, 1] = atom.GetTotalNumHs()  # Number of hydrogen atoms
        atom_features[idx, 2] = atom.GetFormalCharge()  # Formal charge

    # Convert to PyTorch tensors
    atom_features = torch.tensor(atom_features, dtype=torch.float)

    # Create edge_index using the adjacency matrix
    edge_index = torch.tensor(np.column_stack(np.where(adjacency_matrix)), dtype=torch.long)

    # Create PyTorch Geometric data object
    data = Data(x=atom_features, edge_index=edge_index.t().contiguous())  # Transpose edge_index

    return data

def concatenate_with_cross_attention(emb1, emb2, out_channels, heads):
    if emb1.shape[0] > emb2.shape[0]:
        linear_projection = nn.Linear(emb2.shape[0], emb1.shape[0]).to(device)
        emb2 = linear_projection(emb2.T).T
    elif emb1.shape[0] < emb2.shape[0]:
        linear_projection = nn.Linear(emb1.shape[0], emb2.shape[0]).to(device)
        emb1 = linear_projection(emb1.T).T

    concatenated_emb = torch.cat((emb1, emb2), dim=1)

    multihead_attention = nn.MultiheadAttention(embed_dim=2 * out_channels, num_heads=heads).to(device)
    cross_attended_emb, _ = multihead_attention(concatenated_emb, concatenated_emb, concatenated_emb)

    return cross_attended_emb




