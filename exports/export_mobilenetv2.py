import torch
import torchvision
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from safetensors import safe_open
from safetensors.torch import save_file

model = torchvision.models.mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("testdata/mobilenetv2.pt")

print(model, file=open("testdata/mobilenetv2.txt", "w"))
weights = model.state_dict()
for key, value in weights.items():
    print(key)
save_file(model.state_dict(), "testdata/mobilenetv2.safetensors")