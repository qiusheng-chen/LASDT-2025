import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Spa_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        B, C, _, _ = dots.shape
        # mask = torch.eye(x.size(1)).to(dots.device)
        # mask[:, (x.size(1)) // 2] = 1
        # mask[(x.size(1)) // 2, :] = 1
        #
        # dots = dots.masked_fill(mask==0, -10000)
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Spe_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Spa_Transformer_Block(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Spa_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class Spe_Transformer_Block(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Spe_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
class Spatial_Transformer(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = num_patches
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.con_to_patch = nn.Sequential(
            # nn.LayerNorm(102),
            nn.Conv1d(1, 5, kernel_size=7, padding=1, padding_mode='replicate',
                      stride=2),
            nn.ReLU(),
        )

        ln_dim = ((channels+2-7)//2 +1) * 5
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b n p -> b (n p)'),
            Rearrange('(b c) n -> b c n', c = image_width * image_height),
            nn.LayerNorm(ln_dim),
            nn.Linear(ln_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Spa_Transformer_Block(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))  # 保留的mask数量

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # 从小到大ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  #

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # 保留noise小的所对应的patch
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, img, mask_flag=False, mask_ratio=0.5):
        x = Rearrange('b c h w -> b (h w) c')(img)
        x = Rearrange('b n c -> (b n) c')(x)
        x = x.unsqueeze(1)
        x = self.con_to_patch(x)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        x += self.pos_embedding[:, 1:(n + 1)]
        if mask_flag == True:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_tokens_no_pos = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        cls_tokens  = cls_tokens_no_pos + self.pos_embedding[:, 0]
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        x = self.transformer(x)
        if mask_flag == True:
            return x, mask, ids_restore
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, n // 2 + 1]
        x = self.to_latent(x)
        return x

class Spectral_Transformer(nn.Module):
    def __init__(self, *, image_size, spe_patch_size, num_classes, dim, depth, heads, mlp_dim, stride, pool = 'mean', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.stride = stride
        self.spe_patch_size = spe_patch_size
        assert channels % spe_patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patches = channels // spe_patch_size
        patch_dim = image_width * image_height * spe_patch_size
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.con_to_patch = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=(3, 3, 3), padding=(0, 2, 2), padding_mode='replicate',
                      stride=(2, 2, 2)),
            nn.ReLU()
        )
        conv_embedding_dim = ((image_width + 3) // 2) ** 2 * 3 * ((spe_patch_size - 1) // 2)
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(conv_embedding_dim),
            nn.Linear(conv_embedding_dim, dim),
            nn.LayerNorm(dim),
            # Rearrange('(b c2)')
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Spe_Transformer_Block(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
    def img2patch(self, img):
        b, c, h, w = img.shape
        x = [img[:, i:i+self.spe_patch_size, :, :] for i in range(0, c-self.spe_patch_size + 1, self.stride)]
        x = torch.stack(x, dim=1)
        x = Rearrange('b c1 c2 h w -> b c1 (c2 h w)')(x)
        return x
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))  # 保留的mask数量

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # 从小到大ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  #

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # 保留noise小的所对应的patch
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, img, mask_flag=False, mask_ratio=0.5):
        # x1 = self.img2patch(img)
        x = Rearrange('b (c1 c2) h w -> (b c1) c2 h w', c2=self.spe_patch_size)(img)
        x = x.unsqueeze(1)
        x = self.con_to_patch(x)
        x = Rearrange('(b1 b2) d c h w -> b1 b2 (d c h w)', b2=self.num_patches)(x)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        x += self.pos_embedding[:, 1:(n + 1)]
        if mask_flag == True:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_tokens_no_pos = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        cls_tokens = cls_tokens_no_pos + self.pos_embedding[:, 0]
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        x = self.transformer(x)
        if mask_flag == True:
            return x, mask, ids_restore
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return x