import torch
import torch.nn as nn
from models.hsi_transformer import Spatial_Transformer, Spectral_Transformer

class generator(nn.Module):
    def __init__(self, input_channels, patch_size, spe_dim, spa_dim, num_classes, spe_patch_size, depth, stride, dim_head=128, heads=8, dropout=0, emb_dropout=0):
        super(generator, self).__init__()
        # self.spectral_branch = con_spectral(input_channels=input_channels, patch_size=patch_size, out_channels=out_channels)
        self.spatial_branch = Spatial_Transformer(image_size=patch_size, patch_size=1, num_classes=num_classes,
             dim=spa_dim, depth=depth, heads=heads,
             mlp_dim = spa_dim * 4, pool = 'cls', channels = input_channels,
             dim_head = dim_head, dropout = dropout, emb_dropout = emb_dropout)
        self.spectral_branch = Spectral_Transformer(image_size=patch_size, spe_patch_size=spe_patch_size, num_classes=num_classes,
                                  dim=spe_dim, depth=depth, heads=heads, stride=stride,
                                  mlp_dim=spe_dim * 4, pool='cls', channels=input_channels,
                                  dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)
        self.se = nn.Sequential(
            nn.Linear(in_features=spe_dim + spa_dim,
                      out_features=spe_dim + spa_dim
                      , bias=False))
    def forward(self, x, mask_flag=False, mask_ratio=0.5):
        if mask_flag == True:
            x_spectral, spectral_mask, spectral_idx_restore = self.spectral_branch(x, mask_flag, mask_ratio)
            x_spatial, spatial_mask, spatial_idx_restore = self.spatial_branch(x, mask_flag, mask_ratio)
            return (x_spatial, x_spectral), (spatial_mask, spectral_mask), (spatial_idx_restore, spectral_idx_restore)
        elif mask_flag == False:
            x_spectral = self.spectral_branch(x, mask_flag)
            x_spatial = self.spatial_branch(x, mask_flag)
            x = torch.cat([x_spectral, x_spatial], dim=1)
            se = self.se(x)
            x = x * se
        return x
