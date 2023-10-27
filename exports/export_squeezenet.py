import torch
import torchvision
from torchvision.models.squeezenet import SqueezeNet1_0_Weights,SqueezeNet1_1_Weights
from safetensors import safe_open
from safetensors.torch import save_file

model = torchvision.models.squeezenet1_0(weights = SqueezeNet1_0_Weights.DEFAULT)
model.eval()

print(model, file=open("testdata/squeezenet1_0.txt", "w"))
weights = model.state_dict()
for key, value in weights.items():
    print(key)
save_file(model.state_dict(), "testdata/squeezenet1_0.safetensors")

model = torchvision.models.squeezenet1_1(weights = SqueezeNet1_1_Weights.DEFAULT)
model.eval()

print(model, file=open("testdata/squeezenet1_1.txt", "w"))
weights = model.state_dict()
for key, value in weights.items():
    print(key)
save_file(model.state_dict(), "testdata/squeezenet1_1.safetensors")