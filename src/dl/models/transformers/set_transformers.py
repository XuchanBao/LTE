from spaghettini import quick_register

from torch import nn


@quick_register
class SetTransformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, xs, **kwargs):
        zs = self.encoder(xs)
        outputs = self.decoder(zs)

        return outputs

    @property
    def name(self):
        return "SetTransformer"
