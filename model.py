import BNT_modules as modules
import torch
from torch import nn
import math
from utils import *

class Brain_CLIP(nn.Module):
    def __init__(self, dim_input:int, num_heads:int, encoder_layer_sizes:list, decoder_layer_sizes:list):
        super().__init__()
        self.encoder_layer_sizes = [dim_input] + encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes + [dim_input]
        
        # SC to embed
        self.SC_encoder = modules.Transformer(
            layer_sizes=self.encoder_layer_sizes,
            heads=num_heads,
            batch_first=True
        )
        
        # FC to embed
        self.FC_encoder = modules.Transformer(
            layer_sizes=self.encoder_layer_sizes,
            heads=num_heads,
            batch_first=True
        )
        
        # embed to SC
        self.SC_decoder = modules.Transformer(
            layer_sizes=self.decoder_layer_sizes,
            heads=num_heads,
            batch_first=True
        )
        
        # embed to FC
        self.FC_decoder = modules.Transformer(
            layer_sizes=self.decoder_layer_sizes,
            heads=num_heads,
            batch_first=True
        )
        
        self.input_type = self.SC_encoder.encoder_layers[0].self_attn.in_proj_weight.dtype
    
    def encode_SC(self, image:torch.Tensor):
        return self.SC_encoder(image.type(self.input_type))
    
    def encode_FC(self, image:torch.Tensor):
        return self.FC_encoder(image.type(self.input_type))
    
    def decode_SC(self, embed:torch.Tensor):
        return self.SC_decoder(embed.type(self.input_type))
    
    def decode_FC(self, embed:torch.Tensor):
        return self.FC_decoder(embed.type(self.input_type))
    
    # encoder & decoder for either FC or SC
    def forward(self, batch_SC:torch.Tensor, batch_FC:torch.Tensor):
        # encoders                                                     input: [batch_size, ROI, ROI]
        embed_SC = self.encode_SC(batch_SC)                                 # [batch_size, ROI, embed_dim]
        embed_FC = self.encode_FC(batch_FC)
        
        # decoders (might be better to use normalized embeddings)      input: [batch_size, ROI, embed_dim] 
        pred_SC = self.decode_SC(embed_FC)                                  # [batch_size, ROI, ROI]
        pred_FC = self.decode_FC(embed_SC)
        
        # regeneration
        pred_SC_full = torch.bmm(pred_SC, pred_SC.transpose(1,2))           # [batch_size, ROI, ROI]
        pred_FC_full = torch.bmm(pred_FC, pred_FC.transpose(1,2))
        
        return pred_SC_full, pred_FC_full, embed_SC, embed_FC
    
    def predict_FC(self, SC_data:torch.Tensor):
        embed = self.encode_SC(SC_data)
        pred = self.decode_FC(embed)
        pred_full = torch.bmm(pred, pred.transpose(1,2))
        return pred_full
        
    def predict_SC(self, FC_data:torch.Tensor):
        embed = self.encode_FC(FC_data)
        pred = self.decode_SC(embed)
        pred_full = torch.bmm(pred, pred.transpose(1,2))
        return pred_full
 
        
