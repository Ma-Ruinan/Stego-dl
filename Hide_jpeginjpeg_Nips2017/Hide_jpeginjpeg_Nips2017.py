# encoding: utf-8
# This steganography method is named: Hide_jpeginjpeg_Nips2017
# Created by: Ruinan Ma
# Created time��2022/09/28
"""
    This is a PyTorch implementation of image steganography via deep learning, 
    which is released in paper "Hiding Images in Plain Sight: Deep Steganography"
    [https://proceedings.neurips.cc/paper/2017/hash/838e8afb1ca34354ac209f53d90c3a43-Abstract.html]
"""
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
          
class BaseCodec():
    def encode(self, *args, **kwargs):
        None
    def decode(self, *args, **kwargs):
        None
        
        
class Hide_jpeginjpeg_Nips2017(BaseCodec):
    def __init__(self):
        super().__init__()
        self.Hnet_pth = "./checkPoint/netH.pth"
        self.Rnet_pth = "./checkPoint/netR.pth"
        self.Hnet, self.Rnet = self.net_init(self.Hnet_pth, self.Rnet_pth)
        self.Hnet.eval()
        self.Rnet.eval()
        self.Hnet.zero_grad()
        self.Rnet.zero_grad()
        self.transform = transforms.Compose([
                    transforms.Resize([256, 256]),
                    transforms.ToTensor()])
        if torch.cuda.is_available():
                print("2.You are running 'Hide_jpegtojpeg_Nips2017' with GPU.")
        else:
            print("2.You are running 'Hide_jpegtojpeg_Nips2017' with CPU.")
        
    def net_init(self, Hnet_pth, Rnet_pth):
        Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
        Hnet.cuda()
        Hnet.apply(weights_init)
        Hnet.load_state_dict(torch.load(Hnet_pth))
        Rnet = RevealNet(output_function=nn.Sigmoid)
        Rnet.cuda()
        Rnet.apply(weights_init)
        Rnet.load_state_dict(torch.load(Rnet_pth))
        print("1.Hnet and Rnet load pre_trained model successfully.")
        return Hnet, Rnet  
        
    def encode(self, cover, stego):
        print("Encoding...")
        cover = self.transform(cover)  # cover.size()-->torch.Size([3, 256, 256])
        stego = self.transform(stego)
        cover = cover.unsqueeze(dim=0)  # cover.size()-->torch.Size([1, 3, 256, 256])
        stego = stego.unsqueeze(dim=0)
        concat_img = torch.cat([cover, stego], dim=1)  # concat_img.size()-->torch.Size([1, 6, 256, 256])
        if torch.cuda.is_available():
            cover = cover.cuda()
            stego = stego.cuda()
            concat_img = concat_img.cuda()
        container_img = self.Hnet(concat_img)  # container_img.size()-->torch.Size([1, 3, 256, 256])
        if not os.path.isdir("./result-pic"):
            os.mkdir("./result-pic")
        torchvision.utils.save_image(container_img, "./result-pic/container.png")
        
    def decode(self, container):
        print("Dncoding...")
        container = self.transform(container)
        container = container.unsqueeze(dim=0)
        if torch.cuda.is_available():
            container = container.cuda()
        reveal_img = self.Rnet(container)
        if not os.path.isdir("./result-pic"):
            os.mkdir("./result-pic")
        torchvision.utils.save_image(reveal_img, "./result-pic/reveal.png")
        

cover_img = Image.open("./test-pic/1.JPEG")
stego_img = Image.open("./test-pic/2.JPEG")
stego = Hide_jpeginjpeg_Nips2017()
stego.encode(cover=cover_img, stego=stego_img)
container_img = Image.open("./result-pic/container.png")
stego.decode(container=container_img)