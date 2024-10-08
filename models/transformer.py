import torch
import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder

class Transformer(nn.Module):
    # def __init__(self, num_user, max_degree_user, num_item, max_degree_item, d_model, d_ffn, num_heads, dropout, num_layers_enc, num_layers_dec):
    def __init__(self, num_user, max_degree_user, max_spd_value, num_item, max_degree_item, d_model, d_ffn, num_heads, dropout, num_layers_enc, num_layers_dec):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_user=num_user,
            max_degree=max_degree_user,
            max_spd_value=max_spd_value,
            d_model=d_model,
            d_ffn=d_ffn,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers_enc
        )

        self.decoder = Decoder(
            num_item=num_item,
            max_degree=max_degree_item,
            d_model=d_model,
            d_ffn=d_ffn,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers_dec
        )
    
    def forward(self, batched_data):
        enc_output, enc_loss = self.encoder(batched_data)
        # print(f"############### Enc end... {enc_output.shape} and {src_mask.shape} ###############")
        output, dec_loss = self.decoder(batched_data, enc_output)

        # [batch_size, seq_leng_item, seq_len_user]
        # ==> [batch_size, seq_len_user, seq_len_item]
        return output.permute(0, 2, 1), enc_loss ,dec_loss#(enc_loss + dec_loss) / 2