"""
Implementation taken from https://github.com/juho-lee/set_transformer/blob/master/models.py
"""
from spaghettini import quick_register

import torch
import torch.nn as nn


@quick_register
class DenseDeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, num_encoder_blocks, num_decoder_layers, dim_hidden=512):
        super(DenseDeepSet, self).__init__()
        assert num_encoder_blocks >= 2
        assert num_decoder_layers >= 2
        self.num_outputs = num_outputs
        self.dim_output = dim_output

        # Construct Deep Set Encoder.
        deep_set_blocks = list()
        deep_set_blocks.append(
            DeepSetEncoderBlock(dim_input=dim_input, dim_output=dim_hidden, dim_dec_output=int(0.25 * dim_hidden),
                                dim_hidden=dim_hidden, add_output_nonlinearity=True))
        if num_encoder_blocks > 2:
            for i in range(num_encoder_blocks - 2):
                deep_set_blocks.append(
                    DeepSetEncoderBlock(dim_input=dim_hidden, dim_output=dim_hidden,
                                        dim_dec_output=int(0.25 * dim_hidden), dim_hidden=dim_hidden,
                                        add_output_nonlinearity=True))
        deep_set_blocks.append(
            DeepSetEncoderBlock(dim_input=dim_hidden, dim_output=dim_hidden, dim_dec_output=int(0.25 * dim_hidden),
                                dim_hidden=dim_hidden, add_output_nonlinearity=False))

        self.enc = nn.Sequential(*deep_set_blocks)

        # Construct Deep set decoder.
        deep_set_decoder_layers = list()
        for i in range(num_decoder_layers - 1):
            deep_set_decoder_layers.append(nn.Linear(dim_hidden, dim_hidden))
            deep_set_decoder_layers.append(nn.ReLU())
        deep_set_decoder_layers.append(nn.Linear(dim_hidden, num_outputs * dim_output))

        self.dec = nn.Sequential(*deep_set_decoder_layers)

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X.squeeze()


@quick_register
class DeepSetEncoderBlock(nn.Module):
    def __init__(self, dim_input, dim_output, dim_dec_output, dim_hidden=512, add_output_nonlinearity=True):
        super(DeepSetEncoderBlock, self).__init__()
        self.dim_output = dim_output
        self.dim_enc_output = dim_output - dim_dec_output
        self.dim_dec_output = dim_dec_output
        self.add_output_nonlinearity = add_output_nonlinearity

        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, self.dim_enc_output))
        self.dec = nn.Sequential(
            nn.Linear(self.dim_enc_output, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_dec_output))

    def forward(self, X):
        num_inputs = X.shape[1]
        X_enc = self.enc(X)
        X_dec = self.dec(X_enc.mean(-2))

        # Reshape and repeat the decoder output so that it can be appended to the encoder output.
        X_dec = X_dec[:, None, :].repeat(1, num_inputs, 1)

        X = torch.cat((X_enc, X_dec), dim=2)

        if self.add_output_nonlinearity:
            X = torch.relu(X)

        return X


@quick_register
class DeepSetOriginal(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=512):
        super(DeepSetOriginal, self).__init__()
        self.dim_input = dim_input
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, num_outputs * dim_output))

    def forward(self, X, **kwargs):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X.squeeze()

    @property
    def name(self):
        return "DeepSetOriginal"


if __name__ == "__main__":
    """
    Run from root. 
    python -m src.dl.models.deepsets.deepsets
    """
    test_num = 2

    if test_num == 0:
        # ____ Pass random inputs through DeepSetOriginal. ____
        bs_, num_input_, dim_input_, num_outputs_, dim_output_, dim_hidden_ = 5, 10, 16, 2, 10, 512
        x = torch.rand(size=(bs_, num_input_, dim_input_))
        ds = DeepSetOriginal(dim_input=dim_input_, num_outputs=num_outputs_, dim_output=dim_output_,
                             dim_hidden=dim_hidden_)

        y = ds(x)

    if test_num == 1:
        # ____ Pass random inputs through DeepSetBlock. ____
        bs_, num_input_, dim_input_, dim_output_, dim_dec_output_, dim_hidden_ = 5, 10, 16, 512, 128, 512
        x = torch.rand(size=(bs_, num_input_, dim_input_))
        dsb = DeepSetEncoderBlock(dim_input=dim_input_, dim_output=dim_output_, dim_dec_output=dim_dec_output_,
                                  dim_hidden=dim_hidden_)

        y = dsb(x)

    if test_num == 2:
        # ____ Pass random inputs through DenseDeepSet. ____
        bs_, num_input_, dim_input_, dim_output_, dim_dec_output_, dim_hidden_ = 5, 10, 16, 512, 128, 512
        num_outputs_ = 2
        x = torch.rand(size=(bs_, num_input_, dim_input_))
        dds = DenseDeepSet(dim_input=dim_input_, num_outputs=num_outputs_, dim_output=dim_output_,
                           num_encoder_blocks=3, num_decoder_layers=3, dim_hidden=dim_hidden_)
        print(dds)
        y = dds(x)
        breakpoint()
