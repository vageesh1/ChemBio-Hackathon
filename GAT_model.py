import torch 
import torch.nn as nn
import torch.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from helper import smiles_to_graph,separate_compounds,concatenate_with_cross_attention


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, input_size, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.encoding = nn.Embedding(max_len, input_size)

    def forward(self, x):
        x = x.to(device)
        positions = torch.arange(0, x.size(1), device=device).unsqueeze(0)
        positions = positions.expand(x.size(0), -1)  # Expand along the batch dimension
        positions = positions.to(device)
        return x + self.encoding(positions)
    
class DistanceAttentionEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DistanceAttentionEncoder, self).__init__()

        self.embedding = PositionalEncoding(input_size)
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def pairwise_distances(self, x):
        distances = torch.norm(x[:, None, :] - x, dim=-1, p=2)
        return distances

    def forward(self, input_sequence):
        embedded_sequence = self.embedding(input_sequence)
        encoded_sequence = self.encoder(embedded_sequence)
        attention_scores = self.decoder(torch.tanh(encoded_sequence))
        attention_weights = self.softmax(attention_scores)
        context_vector = torch.sum(encoded_sequence * attention_weights, dim=1)

        return context_vector
    
class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, out_channels, attention_heads=1):
        super().__init__()
        self.W_q = nn.Linear(in_channels, out_channels)
        self.W_k = nn.Linear(in_channels, out_channels)
        self.W_v = nn.Linear(in_channels, out_channels)
        self.attention_heads = attention_heads

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(-1, self.attention_heads, q.size(-1))
        k = k.view(-1, self.attention_heads, k.size(-1))
        v = v.view(-1, self.attention_heads, v.size(-1))

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v).view(x.size(0), -1)

        return attention_output

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, external_attention_heads=None):
        super(GATModel, self).__init__()

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.external_attention = MultiHeadAttention(hidden_channels * heads, hidden_channels, attention_heads=external_attention_heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)
        self.external_attention_heads = external_attention_heads

    def forward(self, data):
        data = data.to(device)
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        if self.external_attention_heads is not None:
            external_attention_output = self.external_attention(x)
            x = torch.cat([x, external_attention_output], dim=-1)

        x = self.conv2(x, edge_index)

        self.distance_attention_encoder = DistanceAttentionEncoder(x.size(1), hidden_size=64).to(device)
        distance_attention_output = self.distance_attention_encoder(x.unsqueeze(0))

        return x
    
def get_cross_attention_output(smiles_string, train_mode=True):
    reactant, product = separate_compounds(smiles_string)
    graph_data_reactant = smiles_to_graph(reactant)
    graph_data_product = smiles_to_graph(product)
    graph_data_reactant = graph_data_reactant.to(device)
    graph_data_product = graph_data_product.to(device)

    in_channels = graph_data_reactant.x.size(1)
    hidden_channels = 64
    out_channels = 32
    heads = 2
    gat_model_reactant = GATModel(in_channels, hidden_channels, out_channels, heads).to(device)

    if train_mode:
        gat_model_reactant.train()
    else:
        gat_model_reactant.eval()

    output_reactant = gat_model_reactant(graph_data_reactant)

    in_channels = graph_data_product.x.size(1)
    hidden_channels = 64
    out_channels = 32
    heads = 2
    gat_model_product = GATModel(in_channels, hidden_channels, out_channels, heads).to(device)

    if train_mode:
        gat_model_product.train()
    else:
        gat_model_product.eval()

    output_product = gat_model_product(graph_data_product)

    out = concatenate_with_cross_attention(output_reactant, output_product, out_channels=32, heads=2)

    return out
