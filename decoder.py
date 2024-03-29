import torch 
import torch.nn as nn  
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from helper import smiles_to_graph,separate_compounds,concatenate_with_cross_attention

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size_condition1, vocab_size_condition2, vocab_size_condition3, d_model=512, nhead=8, num_layers=6, enc_dim=64, max_seq=108):
        super(TransformerDecoder, self).__init__()

        self.vocab_size_condition1 = vocab_size_condition1
        self.vocab_size_condition2 = vocab_size_condition2
        self.vocab_size_condition3 = vocab_size_condition3
        self.d_model = d_model
        self.linear_layer = nn.Linear(vocab_size_condition1, d_model)

        self.embedding1 = nn.Embedding(vocab_size_condition1, d_model)
        self.embedding2 = nn.Embedding(vocab_size_condition2, d_model)
        self.embedding3 = nn.Embedding(vocab_size_condition3, d_model)

        self.transformer_layers = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead),
            num_layers
        )

        self.fc_condition1 = nn.Linear(d_model, vocab_size_condition1)
        self.fc_condition2 = nn.Linear(d_model, vocab_size_condition2)
        self.fc_condition3 = nn.Linear(d_model, vocab_size_condition3)

        self.softmax_condition1 = nn.Softmax(dim=1)
        self.softmax_condition2 = nn.Softmax(dim=1)
        self.softmax_condition3 = nn.Softmax(dim=1)

        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

        self.linear_proj = nn.Linear(max_seq, enc_dim)
        self.max_seq = max_seq

        self.final_linear_layer1 = nn.Linear(d_model, max_seq)
        self.final_linear_layer2 = nn.Linear(d_model, max_seq)
        self.final_linear_layer3 = nn.Linear(d_model, max_seq)

    def forward(self, encoded_input, target_sequence):
        self.emb_cond1 = self.embedding1(target_sequence[0].to(device))
        self.emb_cond2 = self.embedding2(target_sequence[1].to(device))
        self.emb_cond3 = self.embedding3(target_sequence[2].to(device))

        self.comb_emb = self.emb_cond1 + self.emb_cond2 + self.emb_cond3
        self.comb_emb = self.comb_emb.squeeze(0)

        memory = torch.rand(1, self.max_seq, 512).to(device)
        trans_decoder = self.transformer_layers(F.relu(self.comb_emb), memory=memory)
        new_size = (trans_decoder.size(0) * trans_decoder.size(1), -1)
        trans_decoder_2d = trans_decoder.view(*new_size)

        proj_trans_decoder = self.linear_proj(trans_decoder_2d.T).T

        tensor1 = proj_trans_decoder.unsqueeze(1)
        tensor1 = tensor1.permute(1, 2, 0)
        tensor2 = encoded_input.unsqueeze(0)

        attention_weights = F.softmax(torch.bmm(tensor1, tensor2.permute(0, 2, 1)), dim=-1)
        cross_attention_result = torch.bmm(attention_weights, tensor2)
        cross_attention_result = cross_attention_result.squeeze(1)
        new_size = (cross_attention_result.size(0) * cross_attention_result.size(1), -1)
        cross_attention_result = cross_attention_result.view(*new_size)

        final_output1 = self.final_linear_layer1(cross_attention_result.T)
        final_output_summed1 = final_output1.sum(dim=0)

        final_output2 = self.final_linear_layer2(cross_attention_result.T)
        final_output_summed2 = final_output2.sum(dim=0)

        final_output3 = self.final_linear_layer3(cross_attention_result.T)
        final_output_summed3 = final_output3.sum(dim=0)

        return F.relu(final_output_summed1), F.relu(final_output_summed2), F.relu(final_output_summed3)


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size_condition1, vocab_size_condition2, vocab_size_condition3, decoder_d_model=512, decoder_nhead=8, decoder_layers=6, enc_dim=64):
        super(TransformerSeq2Seq, self).__init__()

        self.decoder = TransformerDecoder(vocab_size_condition1, vocab_size_condition2, vocab_size_condition3,
                                         d_model=decoder_d_model, nhead=decoder_nhead, num_layers=decoder_layers, enc_dim=enc_dim)

    def forward(self, input_sequence, targets, train_mode=True):
        self.hidden_state = get_cross_attention_output((input_sequence), train_mode)

        self.decoder = self.decoder.to(device)

        if train_mode:
            self.decoder.train()
        else:
            self.decoder.eval()

        output_probs_condition1, output_probs_condition2, output_probs_condition3 = self.decoder(self.hidden_state, targets)

        return output_probs_condition1,output_probs_condition2,output_probs_condition3
