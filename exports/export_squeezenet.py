import torch
import torchvision
from torchvision.models.squeezenet import SqueezeNet1_0_Weights
from safetensors import safe_open
from safetensors.torch import save_file

model = torchvision.models.squeezenet1_0(weights = SqueezeNet1_0_Weights.DEFAULT)
model.eval()
# example = torch.rand(1, 3, 224, 224)
# traced_script_module = torch.jit.trace(model, example)
# traced_script_module.save("testdata/squeezenet1_0.pt")

print(model, file=open("testdata/squeezenet1_0.txt", "w"))
weights = model.state_dict()
for key, value in weights.items():
    print(key)
save_file(model.state_dict(), "testdata/squeezenet1_0.safetensors")