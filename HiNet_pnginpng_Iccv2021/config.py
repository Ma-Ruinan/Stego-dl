# Super parameters
device_ids = [0]
log10_lr = -4.5
lr = 10 ** log10_lr
betas = (0.5, 0.999)
weight_decay = 1e-5
weight_step = 1000
gamma = 0.5
channels_in = 3
# cropsize_val can change by youself
cropsize_val = 256
# clamp is in invblock.py
clamp = 2.0
# init_scale is in model.py
init_scale = 0.01