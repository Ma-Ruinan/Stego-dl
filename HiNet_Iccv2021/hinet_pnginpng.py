import torch
import torch.nn
import torch.optim
import torchvision
from torchvision import transforms
from PIL import Image
import config as c
from model import *
import Unet_common as common

def load(name, net, optim):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')
        
def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()
    return noise


class HiNet_Iccv2021():
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.para_pth: str = "./model/model.pt"
        self.transform = transforms.Compose(
            [transforms.CenterCrop(c.cropsize_val),transforms.ToTensor()]
        )
        self.dwt = common.DWT()
        self.iwt = common.IWT()
        self.HiNet = self.net_init()
        if verbose:
            if torch.cuda.is_available():
                print("Running HiNet_Iccv2021 with GPU.")
            else:
                print("Running HiNet_Iccv2021 with CPU.")
                
    def net_init(self):
        net = Model()
        net.cuda()
        init_model(net)
        net = torch.nn.DataParallel(net, device_ids=c.device_ids)
        params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
        optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
        weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)
        load(self.para_pth, net, optim)
        net.eval()
        if self.verbose:
            print("HiNet loads pre_trained model successfully.")
        return net
        
    
    def encode(self, carrier: Image.Image, payload: Image.Image) -> torch.Tensor: 
        if self.verbose:
            print("Encoding...")
        carrier = self.transform(carrier)
        payload = self.transform(payload)
        # carrier.size()=payload.size()-->torch.Size([1, 3, 256, 256])
        carrier = carrier.unsqueeze(dim=0)
        payload = payload.unsqueeze(dim=0)
 
        # carrier_input.size()=payload_input.size()-->torch.Size([1, 12, 128, 128])
        carrier_input = self.dwt(carrier)
        payload_input = self.dwt(payload)
        
        # input_img.size()-->torch.Size([1, 24, 128, 128])
        input_img = torch.cat([carrier_input, payload_input], dim=1)
        
        if torch.cuda.is_available():
            input_img = input_img.cuda()
        
        # output.size()-->torch.Size([1, 24, 128, 128])
        output = self.HiNet(input_img)
        # output_steg.size()-->torch.Size([1, 12, 128, 128])
        output_steg = output.narrow(1, 0, 4 * c.channels_in)
        # output_z.size()-->torch.Size([1, 12, 128, 128])
        output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
        container = self.iwt(output_steg)
        backward_z = gauss_noise(output_z.shape)
        return container
    
    def decode(self, container: Image.Image) -> torch.Tensor:
        if self.verbose:
            print("Decoding...")
        container = self.transform(container)
        container = container.unsqueeze(dim=0)
        
        if torch.cuda.is_available():
            container = container.cuda()
        
        container = self.dwt(container)
        backward_z = gauss_noise(container.shape)
        output_rev = torch.cat([container, backward_z], dim=1)
        bacward_img = self.HiNet(output_rev, rev=True)
        secret_rev = bacward_img.narrow(1, 4 * c.channels_in, bacward_img.shape[1] - 4 * c.channels_in)
        reveal_img = self.iwt(secret_rev)
        return reveal_img
        

# For test(encode)
carrier = Image.open("./test-pic/fox.png")
payload = Image.open("./test-pic/hulk.png") 
stego = HiNet_Iccv2021(verbose=True)
container = stego.encode(carrier, payload)
torchvision.utils.save_image(container, "./test-pic/container.png")
# For test(decode)
container = Image.open("./test-pic/container.png")
reveal_img = stego.decode(container)
torchvision.utils.save_image(reveal_img, "./test-pic/reveal_img.png")


        