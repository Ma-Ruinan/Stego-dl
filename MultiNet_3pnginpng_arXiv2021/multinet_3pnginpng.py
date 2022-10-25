"""
This steganography method is named: `MultiNet3PNGinPNGarxiv2021`

* Created by: Ruinan Ma
* Created time: 2022/10/27

This is a PyTorch implementation of image steganography via deep learning, which is
released in paper - Multi-Image Steganography Using Deep Neural Networks
https://arxiv.org/pdf/2101.00350.pdf
"""
import re
from PIL import Image
import torch
import torchvision
from torchvision import transforms

from e_model import PrepNetwork1, PrepNetwork2, PrepNetwork3, HidingNetwork
from r_model import RevealNetwork1, RevealNetwork2, RevealNetwork3
from module import Encoder, Decoder, SteganoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiNet3PNGinPNGarxiv2021:
    def __init__(
        self,
        para_path: str = "./ckpt/model_1000.pkl",
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.para_path = para_path
        self.model = torch.load(self.para_path, map_location=device)
        self.transform = transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor()]
        )
        if verbose:
            if torch.cuda.is_available():
                print("Running MultiNet with GPU.")
            else:
                print("Running MultiNet with CPU.")

    def encode(self, carrier, payload) -> None:
        raise NotImplementedError("Use encode_multiple instead.")

    def decode(self, container) -> None:
        raise NotImplementedError("Use decode_multiple instead.")

    def encode_multiple(
        self,
        carrier: Image.Image,
        payload1: Image.Image,
        payload2: Image.Image,
        payload3: Image.Image,
    ) -> torch.Tensor:
        if self.verbose:
            print("Encoding...")
        carrier = self.transform(carrier)
        payload1 = self.transform(payload1)
        payload2 = self.transform(payload2)
        payload3 = self.transform(payload3)

        carrier = carrier.unsqueeze(dim=0)
        payload1 = payload1.unsqueeze(dim=0)
        payload2 = payload2.unsqueeze(dim=0)
        payload3 = payload3.unsqueeze(dim=0)

        carrier = carrier.to(device)
        payload1 = payload1.to(device)
        payload2 = payload2.to(device)
        payload3 = payload3.to(device)

        container = self.model.encoder(carrier, payload1, payload2, payload3)

        return container

    def decoder_multiple(self, container: Image.Image) -> torch.Tensor:
        if self.verbose:
            print("Decoding...")
        container = self.transform(container)
        container = container.unsqueeze(dim=0)
        container = container.to(device)

        reveal_payload1, reveal_payload2, reveal_payload3 = self.model.decoder(container)

        return reveal_payload1, reveal_payload2, reveal_payload3


# For test
# Encode
carrier = Image.open("./test-pic/fox.png")
payload1 = Image.open("./test-pic/leopard.png")
payload2 = Image.open("./test-pic/monkey.png")
payload3 = Image.open("./test-pic/panda.png")
stego = MultiNet3PNGinPNGarxiv2021(verbose=True)
container = stego.encode_multiple(carrier, payload1, payload2, payload3)
torchvision.utils.save_image(container, "./result-pic/container.png")
# Decode
container = Image.open("./result-pic/container.png")
reveal_payload1, reveal_payload2, reveal_payload3 = stego.decoder_multiple(container)
torchvision.utils.save_image(reveal_payload1, "./result-pic/reveal_payload1.png")
torchvision.utils.save_image(reveal_payload2, "./result-pic/reveal_payload2.png")
torchvision.utils.save_image(reveal_payload3, "./result-pic/reveal_payload3.png")
