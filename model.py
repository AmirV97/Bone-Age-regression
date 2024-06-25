import torch
import  torch.nn as nn
import transformers


class Regressor_head(nn.Sequential):
    """
    Sequential regressor head
    """
    def __init__(self, in_channels, dropout=0.2):
        super().__init__(
            nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=800, bias=True),
                nn.GELU(),
                nn.LayerNorm(800, elementwise_affine=False),
                nn.Dropout(dropout),
                nn.Linear(in_features=800, out_features=256),
                nn.GELU(),
                nn.LayerNorm(256, elementwise_affine=False),
                nn.Linear(256, 32),
                nn.GELU(),
                nn.LayerNorm(32, elementwise_affine=False),
                nn.Linear(32, 1)
            ))

class BAR_model(nn.Module):
    """
    Bone age regression model.
    Predicts bone age using hand radiographs and sex data.

    Attributes:
        configuration (transformers.Swinv2Config): configuration for the backbone
        backbone (transformers.Swinv2Model): The SwinV2 tiny backbone
        SEL (nn.Linear): Linear layer for embedding sex information.
        head (nn.Module): Sequential regressor head using extracted image features and sex embedding
        pool_ln (nn.Sequential): Embedding of the model outputs, to be concatenated with SEL outputs

    Methods:
        forward(x, sex):
            Perform a forward pass through the model.
        
            Args:
                x (torch.Tensor): shape [B, 1, H, W]
                sex (torch.Tensor): 0: female, 1: male

            Returns:
                torch.Tensor: shape (batch_size,).
    """

    def __init__(self):
        super().__init__()
        self.configuration = transformers.Swinv2Config(num_channels=1, image_size=500)
        self.backbone = transformers.Swinv2Model(self.configuration)
        self.SEL = nn.Linear(1, 32) #sex embedding layer
        
        self.head = Regressor_head(800, dropout)
        self.pool_ln = nn.Sequential(
            nn.FractionalMaxPool2d(4, output_size=8),
            nn.Conv2d(768, 768, kernel_size=(4), stride=4, groups=768),
            nn.GELU(),
            nn.FractionalMaxPool2d(2, output_size=1),
            nn.Flatten(start_dim=1, end_dim=-1),       
        )

    def forward(self, x, sex):
        sex = self.SEL(sex.unsqueeze(1))
        x = self.backbone(x).last_hidden_state
        x = x.reshape(x.shape[0], 16, 16, x.shape[-1]).permute(0, 3, 1, 2)
        x = self.pool_ln(x)
        x = self.head(torch.concat((sex, x), dim=1))
        return x
