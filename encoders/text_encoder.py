import torch
import transformers
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, 
                 model_name: str,
                 trainable: bool = True) -> None:
        super().__init__()

        self.model = transformers.AutoModel.from_pretrained(model_name)

        for param in self.model.parameters():
            param.requires_grad = trainable
        
        self.target_token_idx = 0
    
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state

        return last_hidden_state[:, self.target_token_idx, :]