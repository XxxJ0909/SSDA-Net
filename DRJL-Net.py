import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange

# SR Network

def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        return tensor.normal_(mean, std).clamp_(min=a, max=b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3,
                 squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):

        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HAB(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=8,
                 shift_size=0,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.conv_scale = conv_scale
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # Conv_X
        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class OCAB(nn.Module):
    # overlapping cross-attention block

    def __init__(self, dim,
                 input_resolution,
                 window_size,
                 overlap_ratio,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=2,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size  # 计算 overlap_win_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size,
                                padding=(self.overlap_win_size - window_size) // 2)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim, dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)
        # print(f"OCAB input shape: {x.shape}")

        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2)  # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1)  # b, h, w, c
        kv = torch.cat((qkv[1], qkv[2]), dim=1)  # b, 2*c, h, w

        # partition windows
        q_windows = window_partition(q, self.window_size)  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        kv_windows = self.unfold(kv)  # b, c*w*w, nw
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c,
                               owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()  # 2, nw*b, ow*ow, c
        k_windows, v_windows = kv_windows[0], kv_windows[1]  # nw*b, ow*ow, c

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)  # nw*b, nH, nq, d
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)  # nw*b, nH, n, d是头的维数
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)  # nw*b, nH, n, d

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size,
            -1)  # ws*ws, wse*wse, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        x = x.view(b, h * w, self.dim)

        x = self.proj(x) + shortcut

        x = x + self.mlp(self.norm2(x))
        return x


class AttenBlocks(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            HAB(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # OCAB
        self.overlap_attn = OCAB(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, params):
        for blk in self.blocks:
            x = blk(x, x_size, params['rpi_sa'], params['attn_mask'])

        x = self.overlap_attn(x, x_size, params['rpi_oca'])

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RHAG(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=128,
                 patch_size=4,
                 resi_connection='1conv'):
        super(RHAG, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = AttenBlocks(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size, params):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x


class PatchEmbed(nn.Module):

    def __init__(self, img_size=128, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x


class Upsample(nn.Sequential):

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class HAT(nn.Module):
    def __init__(self,
                 img_size=128,  
                 patch_size=1,
                 in_chans=3,
                 embed_dim=32,  
                 depths=(6, 6, 6, 6),
                 num_heads=(8, 8, 8, 8),  
                 window_size=8,  
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.1,  
                 overlap_ratio=0.5,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,  
                 img_range=1.,
                 upsampler='pixelshuffle',
                 resi_connection='1conv',
                 **kwargs):
        super(HAT, self).__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64  
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.5, 0.5, 0.5)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        
        relative_position_index_SA = self.calculate_rpi_sa()
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)

       
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)  

       
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

    
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

   
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

       
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHAG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],  
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,  
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv_after_body = nn.Identity()

  
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),  
                nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)  
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self):
        # calculate relative position index for SA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2

        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww  [64,64]
        return relative_position_index

    def calculate_rpi_oca(self):
        # calculate relative position index for OCA
        window_size_ori = self.window_size  # 8
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)  # 12

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws  2 8 8
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws 2 64

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse  2 12 12
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse  2 144

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]  # 2, ws*ws, wse*wse 2 64 144

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2 64 144

        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)  # ws*ws, wse*wse 64 144
        return relative_position_index

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        # for h in h_slices:
        #     for w in w_slices:
        #         img_mask[:, h, w, :] = cnt
        #         cnt += 1
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])

        # Calculate attention mask and relative position index in advance to speed up inference.
        # The original code is very time-consuming for large window size.
        attn_mask = self.calculate_mask(x_size).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA,
                  'rpi_oca': self.relative_position_index_OCA}

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        
        e1 = self.layers[0](x, x_size, params)
        e2 = self.layers[1](e1, x_size, params)
        e3 = self.layers[2](e2, x_size, params)
        e4 = self.layers[3](e3, x_size, params)

        x = self.norm(e4)  
        x = self.patch_unembed(x, x_size)

        return x, e1, e2, e3, e4

    def forward(self, x, y):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
           
            x = self.conv_first(x)  # [B,3,128,128] -> [B,32,128,128]

           
            features_x, e1_1, e1_2, e1_3, e1_4 = self.forward_features(x)
            res_x = self.conv_after_body(features_x) 
            x = res_x + x  

           
            x = self.conv_before_upsample(x)  
            x = self.upsample(x)  
            H1 = self.conv_last(x)  

        self.mean = self.mean.type_as(y)
        y = (y - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            

            y = F.interpolate(y, scale_factor=0.5, mode='bicubic', align_corners=True)

            y = self.conv_first(y)  

            
            features_y, e2_1, e2_2, e2_3, e2_4 = self.forward_features(y)
            res_y = self.conv_after_body(features_y) 
            y = res_y + y 

            
            y = self.conv_before_upsample(y)  
            y = self.upsample(y)  
            H2 = self.conv_last(y) 

        return H1, H2, e1_1, e1_2, e1_3, e1_4, e2_1, e2_2, e2_3, e2_4

#CD Network
def gram_matrix(feat):
    b, c, h, w = feat.size()
    feat = feat.view(b, c, h * w)  # [b, c, h*w]
    feat_t = feat.transpose(1, 2)  # [b, h*w, c]
    gram = torch.bmm(feat, feat_t) / (c * h * w)  # [b, c, c]
    return gram

#SpeAB
class lrl_block(nn.Module):
    def __init__(self, in_channels, out_channels, act_layer=nn.ReLU, scale_factor=2):
        super(lrl_block, self).__init__()
        self.scale_factor = scale_factor
        self.act_layer = act_layer
        self.relu = self.act_layer()
        self.conv_e1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_e1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_e1_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_e1_4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.conv_e2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_e2_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_e2_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_e2_4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.bn_e1_1 = nn.BatchNorm2d(out_channels)
        self.bn_e1_2 = nn.BatchNorm2d(out_channels)
        self.bn_e1_3 = nn.BatchNorm2d(out_channels)
        self.bn_e1_4 = nn.BatchNorm2d(out_channels)
        self.bn_e2_1 = nn.BatchNorm2d(out_channels)
        self.bn_e2_2 = nn.BatchNorm2d(out_channels)
        self.bn_e2_3 = nn.BatchNorm2d(out_channels)
        self.bn_e2_4 = nn.BatchNorm2d(out_channels)

        self.conv_stacked_output1 = nn.Conv2d(out_channels * 2, out_channels * scale_factor ** 2, kernel_size=3,
                                              padding=1)
        self.conv_stacked_output2 = nn.Conv2d(out_channels * 2, out_channels * scale_factor ** 2, kernel_size=3,
                                              padding=1)
        self.conv_stacked_output3 = nn.Conv2d(out_channels * 2, out_channels * scale_factor ** 2, kernel_size=3,
                                              padding=1)
        self.conv_stacked_output4 = nn.Conv2d(out_channels * 2, out_channels * scale_factor ** 2, kernel_size=3,
                                              padding=1)

        self.pixel_shuffle1 = nn.PixelShuffle(self.scale_factor)
        self.pixel_shuffle2 = nn.PixelShuffle(self.scale_factor)
        self.pixel_shuffle3 = nn.PixelShuffle(self.scale_factor)
        self.pixel_shuffle4 = nn.PixelShuffle(self.scale_factor)

        self.conv_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv_3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv_4 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.bn_3 = nn.BatchNorm2d(out_channels)
        self.bn_4 = nn.BatchNorm2d(out_channels)

        self.downsample1 = nn.Sequential(nn.Conv2d(out_channels, 64, kernel_size=1),  
                                         nn.MaxPool2d(kernel_size=2, stride=2))  

        self.downsample2 = nn.Sequential(nn.Conv2d(out_channels, 128, kernel_size=1),  
                                         nn.MaxPool2d(kernel_size=2, stride=2), nn.MaxPool2d(kernel_size=2,
                                                                                             stride=2))  
        self.downsample3 = nn.Sequential(nn.Conv2d(out_channels, 256, kernel_size=1),  
                                         nn.MaxPool2d(kernel_size=2, stride=2),  
                                         nn.MaxPool2d(kernel_size=2, stride=2),  
                                         nn.MaxPool2d(kernel_size=2, stride=2))  
        self.downsample4 = nn.Sequential(nn.Conv2d(out_channels, 512, kernel_size=1),  
                                         nn.MaxPool2d(kernel_size=2, stride=2),  
                                         nn.MaxPool2d(kernel_size=2, stride=2),  
                                         nn.MaxPool2d(kernel_size=2, stride=2),  
                                         nn.MaxPool2d(kernel_size=2, stride=2))  




    def forward(self, e1_1, e1_2, e1_3, e1_4, e2_1, e2_2, e2_3, e2_4):
        b, hw, c = e1_1.size()
        h = w = int(math.sqrt(hw))

        e1_1 = e1_1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        e1_2 = e1_2.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]
        e1_3 = e1_3.reshape(b, h, w, c).permute(0, 3, 1, 2)
        e1_4 = e1_4.reshape(b, h, w, c).permute(0, 3, 1, 2)
        

        e2_1 = e2_1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        e2_2 = e2_2.reshape(b, h, w, c).permute(0, 3, 1, 2)
        e2_3 = e2_3.reshape(b, h, w, c).permute(0, 3, 1, 2)
        e2_4 = e2_4.reshape(b, h, w, c).permute(0, 3, 1, 2)


        r1_1 = self.relu(self.bn_e1_1(self.conv_e1_1(e1_1)))
        r1_2 = self.relu(self.bn_e1_2(self.conv_e1_2(e1_2)))
        r1_3 = self.relu(self.bn_e1_3(self.conv_e1_3(e1_3)))
        r1_4 = self.relu(self.bn_e1_4(self.conv_e1_4(e1_4)))
        

        r2_1 = self.relu(self.bn_e2_1(self.conv_e2_1(e2_1)))
        r2_2 = self.relu(self.bn_e2_2(self.conv_e2_2(e2_2)))
        r2_3 = self.relu(self.bn_e2_3(self.conv_e2_3(e2_3)))
        r2_4 = self.relu(self.bn_e2_4(self.conv_e2_4(e2_4)))

        g1_1 = gram_matrix(r1_1)
        g1_2 = gram_matrix(r1_2)
        g1_3 = gram_matrix(r1_3)
        g1_4 = gram_matrix(r1_4)


        g2_1 = gram_matrix(r2_1)
        g2_2 = gram_matrix(r2_2)
        g2_3 = gram_matrix(r2_3)
        g2_4 = gram_matrix(r2_4)
       

        d1 = torch.abs(g1_1 - g2_1) * 0.5
        d2 = torch.abs(g1_2 - g2_2) * 0.5
        d3 = torch.abs(g1_3 - g2_3) * 0.5
        d4 = torch.abs(g1_4 - g2_4) * 0.5
        

        r1_1_flat = r1_1.view(b, r1_1.size(1), -1)  
        r1_2_flat = r1_2.view(b, r1_2.size(1), -1)
        r1_3_flat = r1_3.view(b, r1_3.size(1), -1)
        r1_4_flat = r1_4.view(b, r1_4.size(1), -1)
        

        r2_1_flat = r2_1.view(b, r2_1.size(1), -1)
        r2_2_flat = r2_2.view(b, r2_2.size(1), -1)
        r2_3_flat = r2_3.view(b, r2_3.size(1), -1)
        r2_4_flat = r2_4.view(b, r2_4.size(1), -1)

        p1_1 = torch.bmm(d1, r1_1_flat)  
        p1_1 = p1_1.view(b, -1, h, w)  
        p1_1 = torch.abs(r1_1 - p1_1)

        p1_2 = torch.bmm(d2, r1_2_flat).view(b, -1, h, w)
        p1_2 = torch.abs(r1_2 - p1_2)

        p1_3 = torch.bmm(d3, r1_3_flat).view(b, -1, h, w)
        p1_3 = torch.abs(r1_3 - p1_3)

        p1_4 = torch.bmm(d4, r1_4_flat).view(b, -1, h, w)
        p1_4 = torch.abs(r1_4 - p1_4)

        p2_1 = torch.bmm(d1, r2_1_flat).view(b, -1, h, w)
        p2_1 = r2_1 + p2_1

        p2_2 = torch.bmm(d2, r2_2_flat).view(b, -1, h, w)
        p2_2 = r2_2 + p2_2

        p2_3 = torch.bmm(d3, r2_3_flat).view(b, -1, h, w)
        p2_3 = r2_3 + p2_3

        p2_4 = torch.bmm(d4, r2_4_flat).view(b, -1, h, w)
        p2_4 = r2_4 + p2_4

        output1 = torch.cat((p1_1, p2_1), dim=1)
        output2 = torch.cat((p1_2, p2_2), dim=1)
        output3 = torch.cat((p1_3, p2_3), dim=1)
        output4 = torch.cat((p1_4, p2_4), dim=1)

        output1 = self.pixel_shuffle1(self.conv_stacked_output1(output1))
        output2 = self.pixel_shuffle2(self.conv_stacked_output2(output2))
        output3 = self.pixel_shuffle3(self.conv_stacked_output3(output3))
        output4 = self.pixel_shuffle4(self.conv_stacked_output4(output4))

        s1 = self.relu(self.bn_1(self.conv_1(output1)))
        s2 = self.relu(self.bn_2(self.conv_2(output2)))
        s3 = self.relu(self.bn_3(self.conv_3(output3)))
        s4 = self.relu(self.bn_4(self.conv_4(output4)))

       

        s1 = self.downsample1(s1)  
        s2 = self.downsample2(s2)  
        s3 = self.downsample3(s3)  
        s4 = self.downsample4(s4) 


        return s1, s2, s3, s4

class PatchEmbedCD(nn.Module):  # H1，H2 and SpeAB output
    def __init__(self, img_size, patch_size=None, in_c=3, embed_dim=None, norm_layer=None):  
        super().__init__()
        img_size = (img_size, img_size)
        if patch_size is None:
            if img_size[0] == 128:
                patch_size = 4
            elif img_size[0] == 64:
                patch_size = 2
            elif img_size[0] == 32:
                patch_size = 1
            elif img_size[0] == 16:
                patch_size = 1
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
 
        if embed_dim is None:
            if img_size[0] == 128:
                embed_dim = 1024  
            elif img_size[0] == 64:
                embed_dim = 512  
            elif img_size[0] == 32:
                embed_dim = 256  
            elif img_size[0] == 16:
                embed_dim = 256  
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

# MSA and CRA
class Transformer(nn.Module):
    def __init__(self, img_size, dim=768, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Transformer, self).__init__()
        
        if img_size == 128:
            dim = 1024  
        elif img_size == 64:
            dim = 512  
        elif img_size == 32:
            dim = 256  
        elif img_size == 16:
            dim = 256  
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q1_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k1_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v1_proj = nn.Linear(dim, dim, bias=qkv_bias)
       
        self.vl_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kl_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q2_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k2_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v3_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        
        if img_size == 128:
            in_c = 64  
        elif img_size == 64:
            in_c = 128  
        elif img_size == 32:
            in_c = 256 
        elif img_size == 16:
            in_c = 512

        
        self.PatchEmbed1 = PatchEmbedCD(img_size=img_size, in_c=in_c, patch_size=None, embed_dim=None)
        self.PatchEmbed2 = PatchEmbedCD(img_size=img_size, in_c=in_c, patch_size=None, embed_dim=None)
        self.PatchEmbed3 = PatchEmbedCD(img_size=img_size, in_c=in_c, patch_size=None, embed_dim=None)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)

    def forward(self, x, y, z):  # x:h1 y:s1234 z:h2
        x = self.ln1(self.PatchEmbed1(x))
        y = self.ln2(self.PatchEmbed2(y))
        z = self.ln3(self.PatchEmbed3(z))
        

        q1 = self.q1_proj(x)
        k1 = self.k1_proj(x)
        v1 = self.v1_proj(x)
       
      

        vl = self.vl_proj(y)
        kl = self.kl_proj(y)
        

        q2 = self.q2_proj(z)
        k2 = self.k2_proj(z)
        v2 = self.v3_proj(z)
        
        attn1 = self.compute_attention(q1, v1, k1)  # MSA
        attn1_1 = self.proj_drop(self.proj(attn1))
        

        attn2 = self.compute_attention(q1, vl, kl)  # CRA
        attn2_2 = self.proj_drop(self.proj(attn2))
        

        attn3 = self.compute_attention(q2, vl ,kl)  # CRA
        attn3_3 = self.proj_drop(self.proj(attn3))
        

        attn4 = self.compute_attention(q2, v2, k2)  # MSA
        attn4_4 = self.proj_drop(self.proj(attn4))
        

        attn11 = attn1_1 + attn2_2
        attn22 = attn3_3 + attn4_4

        return attn11, attn22

    def compute_attention(self, q, v, k):
        B, N, C = q.shape
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn = attn @ v
        attn = attn.permute(0, 2, 1, 3).reshape(B, N, C)
        return attn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))
        return x

#SpaAB
class hrl_block(nn.Module):
    def __init__(self, img_size, patch_size, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop_ratio=0.,
                 proj_drop_ratio=0., mlp_ratio=4.):
        super(hrl_block, self).__init__()
        if img_size == 128:  
            dim = 1024  
            in_c = 128
        elif img_size == 64:  
            dim = 512  
            in_c = 256
        elif img_size == 32:  
            dim = 256  
            in_c = 512
        elif img_size == 16:  
            dim = 256  
            in_c = 512

        self.dim = dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_drop_ratio = attn_drop_ratio
        self.proj_drop_ratio = proj_drop_ratio
        self.mlp_ratio = mlp_ratio
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  
        self.custom_transformer = Transformer(img_size=img_size, dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                              qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio,
                                              proj_drop_ratio=proj_drop_ratio)
        self.ln_attn11 = nn.LayerNorm(dim)
        self.ln_attn22 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_attn11 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=proj_drop_ratio)
        self.mlp_attn22 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=proj_drop_ratio)

        self.conv1 = nn.Conv2d(in_c, 32, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, in_c, kernel_size=1)
        self.conv4 = nn.Conv2d(in_c, 32, kernel_size=3, padding=1)  

    def forward(self, x, y, z):
        attn11, attn22 = self.custom_transformer(x, y, z)
        

        
        o1 = attn11 + self.mlp_attn11(self.ln_attn11(attn11))
        o2 = attn22 + self.mlp_attn22(self.ln_attn22(attn22))
       

        
        o1o2 = o1 @ o2.transpose(-2, -1) * self.scale

    
        o_row_softmax = F.softmax(o1o2, dim=-1)
        o_column_softmax = F.softmax(o1o2, dim=-2)

        o_out1 = torch.bmm(o_row_softmax, o1)
        o_out2 = torch.bmm(o2.permute(0, 2, 1), o_column_softmax).permute(0, 2, 1)
        o_hrl = torch.cat((o_out1, o_out2), dim=-1)
       

      
        batch_size, num_patches, embedding_dim = o_hrl.shape

        h = self.img_size
        w = self.img_size

        o_hrl_r = o_hrl.view(batch_size, h, w, -1).permute(0, 3, 1, 2).contiguous()

        f = self.conv4(self.conv3(self.conv2(self.conv1(o_hrl_r))) * o_hrl_r)

        return f

#DRLM
class dualrl_block(nn.Module):
    def __init__(self, in_channels, out_channels, img_size1, img_size2, img_size3, img_size4, patch_size=16, dim=768,
                 num_heads=8, qkv_bias=True,
                 qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0., mlp_ratio=4.):
        super(dualrl_block, self).__init__()

        self.lrl = lrl_block(in_channels=in_channels, out_channels=out_channels)
        self.hrl1 = hrl_block(img_size=img_size1, patch_size=patch_size, dim=dim, num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=proj_drop_ratio,
                              mlp_ratio=mlp_ratio)
        self.hrl2 = hrl_block(img_size=img_size2, patch_size=patch_size, dim=dim, num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio,
                              proj_drop_ratio=proj_drop_ratio, mlp_ratio=mlp_ratio)
        self.hrl3 = hrl_block(img_size=img_size3, patch_size=patch_size, dim=dim, num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio,
                              proj_drop_ratio=proj_drop_ratio, mlp_ratio=mlp_ratio)
        self.hrl4 = hrl_block(img_size=img_size4, patch_size=patch_size, dim=dim, num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio,
                              proj_drop_ratio=proj_drop_ratio, mlp_ratio=mlp_ratio)

    def forward(self, e1_1, e1_2, e1_3, e1_4, e2_1, e2_2, e2_3, e2_4, h1_1, h1_2, h1_3, h1_4, h2_1, h2_2, h2_3, h2_4):
        s1, s2, s3, s4 = self.lrl(e1_1, e1_2, e1_3, e1_4, e2_1, e2_2, e2_3, e2_4)


        f1 = self.hrl1(h1_1, s1, h2_1)
        f2 = self.hrl2(h1_2, s2, h2_2)
        f3 = self.hrl3(h1_3, s3, h2_3)
        f4 = self.hrl4(h1_4, s4, h2_4)


        return f1, f2, f3, f4



class FB(nn.Module):
    def __init__(self, in_channels, out_channels, act_layer=nn.ReLU):
        super(FB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels * 2)
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)


    def forward(self, x, y): 
       
        x = F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=True)
        z = torch.cat((x, y), dim=1)
        z = self.conv2(self.act_layer(self.bn1(self.conv1(z))))
        return z

# Decoder
class MSFD(nn.Module):
    def __init__(self, in_channels, out_channels, act_layer=nn.ReLU):
        super(MSFD, self).__init__()
        self.fb1 = FB(in_channels=in_channels * 2, out_channels=in_channels, act_layer=act_layer)
        self.fb2 = FB(in_channels=in_channels * 2, out_channels=in_channels, act_layer=act_layer)
        self.fb3 = FB(in_channels=in_channels * 2, out_channels=in_channels, act_layer=act_layer)
        self.conv_final = nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=1, stride=1)

    def forward(self, f1, f2, f3, f4):
        f4_upsampled = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=True)
        f4_3 = self.fb1(f4_upsampled, f3)
        f4_3 = f4_3 + f3

        f3_2 = F.interpolate(f4_3, scale_factor=2, mode='bilinear', align_corners=True)
        f3_2 = self.fb1(f3_2, f2)
        f3_2 = f3_2 + f2

        f2_1 = F.interpolate(f3_2, scale_factor=2, mode='bilinear', align_corners=True)
        f2_1 = self.fb1(f2_1, f1)
        f2_1 = f2_1 + f1
        f2_1 = F.interpolate(f2_1, scale_factor=2, mode='bilinear', align_corners=True)
        output = self.conv_final(f2_1)

        return output


class SSDANet(nn.Module):

    def __init__(self,
                 backbone='resnet18',
                 output_stride=32,
                 sr_dim=32,
                 sr_window_size=8,
                 sr_depth=(6, 6, 6, 6),
                 cd_num_heads=8,
                 cd_mlp_ratio=4.,
                 img_size_l1=128,
                 img_size_l2=256,
                 output_nc=2):  
        super(SSDANet, self).__init__()

      
        self.sr_net = HAT(
            dim=sr_dim,
            window_size=sr_window_size,
            depths=sr_depth
        )

       

        from encoder import ResNet18

        if backbone == 'resnet18':
            self.feature_extractor = ResNet18(output_stride, nn.BatchNorm2d, pretrained=True, in_c=3)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")

        self.dualrl = dualrl_block(
            in_channels=32,
            out_channels=32,
            img_size1=img_size_l2 // 2,
            img_size2=img_size_l2 // 4,
            img_size3=img_size_l2 // 8,
            img_size4=img_size_l2 // 16,
            num_heads=8,
            mlp_ratio=cd_mlp_ratio
        )

        self.msfd = MSFD(in_channels=32, out_channels=2, act_layer=nn.ReLU)

    def forward(self, L1, L2):
        """
       
        Args:
            img1: Time1
            img2: Time2
        Returns:
            H1: SR_Time1
            H2: SR_Time2
            e1_1~e1_4, e2_1~e2_4: SR_feature
            cd_map
        """
       
        H1, H2, e1_1, e1_2, e1_3, e1_4, e2_1, e2_2, e2_3, e2_4 = self.sr_net(L1, L2)

       
        h1_1, h1_2, h1_3, h1_4 = self.feature_extractor(H1)
        h2_1, h2_2, h2_3, h2_4 = self.feature_extractor(L2)

        f1,f2,f3,f4 = self.dualrl(e1_1, e1_2, e1_3, e1_4, e2_1, e2_2, e2_3, e2_4,
                                     h1_1, h1_2, h1_3, h1_4, h2_1, h2_2, h2_3, h2_4)

        cd_map = self.msfd(f1,f2,f3,f4)

        return H1, H2, cd_map







