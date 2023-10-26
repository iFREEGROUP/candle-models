# candle-models
caddle 版本经典CV模型结构实现, 支持 resnet、mobilenetv2。

### 特性
支持以下模型
1. resnet系列: resnet18、resnet34、resnet50、resnet101、resnet152
2. mobilenet系列: mobilenetv2

### 导出模型

以 resnet18 为例，导出模型。

```python
import torch
import torchvision
from torchvision.models.resnet import ResNet18_Weights
from safetensors import safe_open
from safetensors.torch import save_file

model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()
# example = torch.rand(1, 3, 224, 224)
# traced_script_module = torch.jit.trace(model, example)
# traced_script_module.save("testdata/resnet18.pt")

# print(model)
weights = model.state_dict()
for key, value in weights.items():
    print(key)# 显示模型各层参数名称
save_file(model.state_dict(), "testdata/resnet18.safetensors")
```

### 推理

```rust
fn main() -> candle_core::Result<()> {
    let model_file = "./testdata/resnet18.safetensors";
    let device = candle_core::Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };


    let image = load_image224("./testdata/mouse.jpg")?;


    let model = resnet18(vb, 1000)?;
    let image = image.unsqueeze(0)?;


    let logits = model.forward(&image)?;
    // println!("{:?}", logits.clone().i(0)?.to_vec1::<f32>());
    let prs = candle_nn::ops::softmax(&logits, D::Minus1)?.i(0)?.to_vec1::<f32>()?;
    
    let mut prs = prs.iter().enumerate().collect::<Vec<_>>();
    prs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for &(category_idx, pr) in prs.iter().take(5) {
        println!("{:24}: {:.2}%", CLASSES[category_idx], 100.0 * pr);
    }
    Ok(())
}

```
结果:
```bash
mouse, computer mouse   : 90.06%
punching bag, punch bag, punching ball, punchball: 4.47%
joystick                : 1.77%
radio, wireless         : 0.45%
vacuum, vacuum cleaner  : 0.20%
test test_resnet18 ... ok
=======

```