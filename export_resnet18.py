import torch
import torchvision
from torchvision.models.resnet import ResNet18_Weights
from safetensors import safe_open
from safetensors.torch import save_file

model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("testdata/resnet18.pt")

print(model)
weights = model.state_dict()
for key, value in weights.items():
    print(key)
save_file(model.state_dict(), "testdata/resnet18.safetensors")