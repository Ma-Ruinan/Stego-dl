"""
This steganography method is named: `DeepMIHMULPNGinPNGTPAMI2022`

* Created by: Ruinan Ma
* Created time: 2022/10/20

This is a PyTorch implementation of image steganography via deep learning, which is
released in paper - DeepMIH: Deep Invertible Network for Multiple Image Hiding
https://ieeexplore.ieee.org/document/9676416
"""
from PIL import Image
import torch
import torch.nn
import torchvision
from torchvision import transforms

import config as c
from unet_common import DWT, IWT
from model import Model_1, Model_2, init_model
from imp_subnet import ImpMapBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gauss_noise(shape):
    noise = torch.zeros(shape).to(device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).to(device)
    return noise


def load(name: str, net: torch.nn.Module) -> None:
    state_dicts = torch.load(name)
    network_state_dict = {
        k: v for k, v in state_dicts["net"].items() if "tmp_var" not in k
    }
    net.load_state_dict(network_state_dict)


class DeepMIHMULPNGinPNGTPAMI2022:
    def __init__(
        self,
        para_path1: str = "./model/model_checkpoint_03000_1.pt",
        para_path2: str = "./model/model_checkpoint_03000_2.pt",
        para_path3: str = "./model/model_checkpoint_03000_3.pt",
        use_img_map: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.use_img_map = use_img_map
        self.para_path1 = para_path1
        self.para_path2 = para_path2
        self.para_path3 = para_path3
        self.dwt = DWT()
        self.iwt = IWT()
        self.transform = transforms.Compose(
            [transforms.CenterCrop(c.cropsize_val), transforms.ToTensor()]
        )
        self.net1, self.net2, self.net3 = self.net_init()
        if verbose:
            if torch.cuda.is_available():
                print("Running DeepMIH with GPU.")
            else:
                print("Running DeepMIH with CPU.")

    def net_init(self):
        net1 = Model_1()
        net2 = Model_2()
        net3 = ImpMapBlock()
        net1.to(device)
        net2.to(device)
        net3.to(device)
        init_model(net1)
        init_model(net2)
        net1 = torch.nn.DataParallel(net1, device_ids=c.device_ids)
        net2 = torch.nn.DataParallel(net2, device_ids=c.device_ids)
        net3 = torch.nn.DataParallel(net3, device_ids=c.device_ids)
        load(self.para_path1, net1)
        load(self.para_path2, net2)
        load(self.para_path3, net3)
        net1.eval()
        net2.eval()
        net3.eval()
        if self.verbose:
            print("Pretrained models load successsfully.")
        return net1, net2, net3

    def encode(
        self, carrier: Image.Image, payload1: Image.Image, payload2: Image.Image
    ) -> torch.Tensor:
        if self.verbose:
            print("Encoding...")
        '''
            prepare
        '''
        # carrier.size()=payload1.size()=payload2.size()-->torch.Size([3, 256, 256])
        carrier = self.transform(carrier)  # type: ignore
        payload1 = self.transform(payload1)  # type: ignore
        payload2 = self.transform(payload2)  # type: ignore
        carrier = carrier.unsqueeze(dim=0)
        payload1 = payload1.unsqueeze(dim=0)
        payload2 = payload2.unsqueeze(dim=0)
        # carrier_dwt.size()=payload1_dwt.size()=payload2_dwt.size()
        # --> torch.Size([1, 12, 128, 128])
        carrier_dwt = self.dwt(carrier).to(device)
        payload1_dwt = self.dwt(payload1).to(device)
        payload2_dwt = self.dwt(payload2).to(device)
        '''
            forward1
        '''
        # input_dwt_1.size()-->torch.Size([1, 24, 128, 128])
        input_dwt_1 = torch.cat((carrier_dwt, payload1_dwt), dim=1)
        # output_dwt_1.size()-->torch.Size([1, 24, 128, 128])
        output_dwt_1 = self.net1(input_dwt_1)
        # output_steg_dwt_1.size()-->torch.Size([1, 12, 128, 128])
        output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)
        # get steg1
        # output_steg_1.size()-->torch.Size([1, 3, 256, 256])
        output_steg_1 = self.iwt(output_steg_dwt_1).to(device)
        '''
            forward2
        '''
        # img_map.size()-->torch.Size([1, 3, 256, 256])
        if self.use_img_map:
            img_map = self.net3(carrier, payload1, output_steg_1)
        else:
            img_map = torch.zeros(carrier.shape).to(device)
        # imp_map_dwt.size()-->torch.Size([1, 12, 128, 128])
        imp_map_dwt = self.dwt(img_map)
        input_dwt_2 = torch.cat((output_steg_dwt_1, imp_map_dwt), dim=1)
        # input_dwt_2.size()-->torch.Size([1, 36, 128, 128])
        input_dwt_2 = torch.cat((input_dwt_2, payload2_dwt), dim=1)
        # output_dwt_2.size()-->torch.Size([1, 36, 128, 128])
        output_dwt_2 = self.net2(input_dwt_2)
        # output_steg_dwt_2.size()-->torch.Size([1, 12, 128, 128])
        output_steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)
        # get steg2
        # output_steg_2.size()-->torch.Size([1, 3, 256, 256])
        output_steg_2 = self.iwt(output_steg_dwt_2).to(device)

        return output_steg_1, output_steg_2

    def decode(self, container1: Image.Image, container2: Image.Image) -> torch.Tensor:
        if self.verbose:
            print("Decoding...")
        '''
            prepare
        '''
        container1, container2 = self.transform(container1), self.transform(container2)
        container1 = container1.unsqueeze(dim=0)
        container2 = container2.unsqueeze(dim=0)
        # container1.size()=container2.size()-->torch.Size([1, 12, 128, 128])
        container1_dwt = self.dwt(container1).to(device)
        container2_dwt = self.dwt(container2).to(device)
        # noise_shape.size()-->torch.Size([1, 24, 128, 128])
        noise_shape = torch.cat((container1_dwt, container2_dwt), dim=1)
        guass1 = gauss_noise(container1_dwt.shape)
        guass2 = gauss_noise(noise_shape.shape)
        '''
            backward2
        '''
        # output_rev_dwt_2.size()-->torch.Size([1, 36, 128, 128])
        output_rev_dwt_2 = torch.cat((container2_dwt, guass2), dim=1)
        # rev_dwt_2.size()-->torch.Size([1, 36, 128, 128])
        rev_dwt_2 = self.net2(output_rev_dwt_2, rev=True)
        # rev_steg_dwt_1.size()=rev_sercet_dwt_1.size()-->torch.Size([1, 36, 128, 128])
        rev_steg_dwt_1 = rev_dwt_2.narrow(1, 0, 4 * c.channels_in)
        rev_sercet_dwt_2 = rev_dwt_2.narrow(1, 4 * c.channels_in, 4 * c.channels_in)
        # if you need middle container, you can export rev_steg_1.
        # rev_steg_1 = self.iwt(rev_steg_dwt_1).to(device)
        rev_sercet_2 = self.iwt(rev_sercet_dwt_2).to(device)
        '''
            backward1
        '''
        # output_rev_dwt_1.size()-->torch.Size([1, 24, 128, 128])
        output_rev_dwt_1 = torch.cat((rev_steg_dwt_1, guass1), dim=1)
        # rev_dwt_1.size()-->torch.Size([1, 24, 128, 128])
        rev_dwt_1 = self.net1(output_rev_dwt_1, rev=True)
        rev_sercet_dwt_1 = rev_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)
        rev_secret_1 = self.iwt(rev_sercet_dwt_1).to(device)

        return rev_sercet_2, rev_secret_1


# For test
# Encode
carrier = Image.open("./test-pic/fox.png")
payload1 = Image.open("./test-pic/hulk.png")
payload2 = Image.open("./test-pic/batman.png")
stego = DeepMIHMULPNGinPNGTPAMI2022(use_img_map=True, verbose=True)
steg1, steg2 = stego.encode(carrier, payload1, payload2)
torchvision.utils.save_image(steg1, "./result-pic/steg1.png")
torchvision.utils.save_image(steg2, "./result-pic/steg2.png")
# Decode
container1 = Image.open("./result-pic/steg1.png")
container2 = Image.open("./result-pic/steg2.png")
reveal_payload2, reveal_payload1 = stego.decode(container1, container2)
torchvision.utils.save_image(reveal_payload2, "./result-pic/reveal_payload2.png")
torchvision.utils.save_image(reveal_payload1, "./result-pic/reveal_payload1.png")
