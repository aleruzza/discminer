import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .nn import timestep_embedding
from .unet import UNetModel


def create_nnmodel(n_param,image_size,num_channels=96,num_res_blocks=3, channel_mult="", mode='cyl', unc=False, drop_skip_connections_ids=[]):
    #num_channels= #128,192
    #num_res_blocks=3 #2,3
    #channel_mult=""
    use_checkpoint=False
    attention_resolutions="16,8"
    num_heads=4
    num_head_channels=-1
    num_heads_upsample=-1
    use_scale_shift_norm=False
    dropout=0
    resblock_updown=False
    use_fp16=False
    use_new_attention_order=False
    
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
    
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
        
    #set number of input or output
    i_ch = 1
    if mode=='mdeco':
        i_ch = 2
    elif mode=='128x128_disc_tri':
        i_ch = 3
        
    o_ch = i_ch*2 if unc else i_ch
    
    return Para2ImUNet(n_param=n_param,
        drop_skip_connections_ids=drop_skip_connections_ids,
        in_channels=i_ch,
        model_channels=num_channels,
        out_channels=o_ch,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown)


class Para2ImUNet(UNetModel):
    """
    A UNetModel that conditions on parameter with linear embedding.

    Expects an extra kwarg `y` of parameter.

    :param n_param: dimension of parameter n_param to expect.
    """

    def __init__(
        self,
        n_param,
        drop_skip_connections_ids,
        *args,
        **kwargs,
    ):
        self.n_param = n_param
        super().__init__(*args, **kwargs)
        self.token_embedding = nn.Linear(n_param, self.model_channels * 4)
        self.drop_skip_connections_ids = drop_skip_connections_ids

    def convert_to_fp16(self):
        super().convert_to_fp16()
        self.token_embedding.to(th.float16)
        self.token_linear.to(th.float16)

    def get_param_emb(self, y):
        assert y is not None

        outputs = self.token_embedding(y)

        return outputs

    def forward(self, x, y=None):
        hs = []
        #emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if y != None:
            emb = self.get_param_emb(y) #previously text_outputs
            #emb = emb + text_outputs.to(emb)

        h = x.type(self.dtype)
        #print(h.shape)
        block_ids = []
        for id, module in enumerate(self.input_blocks):
            h = module(h, emb)
            #print(h.shape)
            hs.append(h)
            block_ids.append(id)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cid = block_ids.pop()
            hss = hs.pop()
            if cid in self.drop_skip_connections_ids:
                h = th.cat([h, hss*0], dim=1)
            else:
                h = th.cat([h, hss], dim=1)
            #print(h.shape)
            h = module(h, emb)
        h = h.type(x.dtype)
        h = self.out(h)
        #print(h.shape)
        return h 
