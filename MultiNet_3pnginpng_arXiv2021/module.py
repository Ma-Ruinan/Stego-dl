import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, reveal_network_1, reveal_network_2, reveal_network_3):
        super(Decoder, self).__init__()
        self.reveal_network_1 = reveal_network_1
        self.reveal_network_2 = reveal_network_2
        self.reveal_network_3 = reveal_network_3

    def forward(self, hidden_image):
        reveal_image_1 = self.reveal_network_1(hidden_image)
        reveal_image_2 = self.reveal_network_2(hidden_image)
        reveal_image_3 = self.reveal_network_3(hidden_image)
        return reveal_image_1, reveal_image_2, reveal_image_3


class Encoder(nn.Module):
    def __init__(self, prep_network_1, prep_network_2, prep_network_3, hiding_network):
        super(Encoder, self).__init__()
        self.prep_network1 = prep_network_1
        self.prep_network2 = prep_network_2
        self.prep_network3 = prep_network_3
        self.hiding_network = hiding_network

    def forward(self, cover_image, secret_image_1, secret_image_2, secret_image_3):
        encoded_secret_image_1 = self.prep_network1(secret_image_1)
        encoded_secret_image_2 = self.prep_network2(secret_image_2)
        encoded_secret_image_3 = self.prep_network3(secret_image_3)

        hidden_image = self.hiding_network(
            encoded_secret_image_1,
            encoded_secret_image_2,
            encoded_secret_image_3,
            cover_image,
        )
        return hidden_image


class SteganoModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(SteganoModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        cover_image,
        secret_image_1,
        secret_image_2,
        secret_image_3,
        hidden_image,
        mode,
    ):
        if mode == "full":
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = False
            hidden_image = self.encoder(
                cover_image, secret_image_1, secret_image_2, secret_image_3
            )
            reveal_image_1, reveal_image_2, reveal_image_3 = self.decoder(hidden_image)
            return hidden_image, reveal_image_1, reveal_image_2, reveal_image_3
        elif mode == "encoder":
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            hidden_image = self.encoder(
                cover_image, secret_image_1, secret_image_2, secret_image_3
            )
            return hidden_image
        elif mode == "decoder":
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = True

            reveal_image1, reveal_image2, reveal_image3 = self.decoder(hidden_image)
            return reveal_image1, reveal_image2, reveal_image3
