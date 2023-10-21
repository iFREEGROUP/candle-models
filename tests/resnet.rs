use candle_core::{ DType, Tensor, Device, Module, D, IndexOp };
use candle_nn::VarBuilder;
use candle_resnet::{resnet18, resnet50};

use crate::imagenet::CLASSES;
mod imagenet;


#[test]
fn test_resnet18() -> candle_core::Result<()> {
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

#[test]
fn test_resnet50() -> candle_core::Result<()> {
    let model_file = "./testdata/resnet50.safetensors";
    let device = candle_core::Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };

    let image = load_image224("./testdata/mouse.jpg")?;

    let model = resnet50(vb, 1000)?;
    let image = image.unsqueeze(0)?;


    let logits = model.forward(&image)?;
    let prs = candle_nn::ops::softmax(&logits, D::Minus1)?.i(0)?.to_vec1::<f32>()?;
    
    let mut prs = prs.iter().enumerate().collect::<Vec<_>>();
    prs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for &(category_idx, pr) in prs.iter().take(5) {
        println!("{:24}: {:.2}%", CLASSES[category_idx], 100.0 * pr);
    }
    Ok(())
}

fn load_image224<P: AsRef<std::path::Path>>(p: P) -> candle_core::Result<Tensor> {
    let img = image::io::Reader
        ::open(p)?
        .decode()
        .map_err(candle_core::Error::wrap)?
        .resize_to_fill(224, 224, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (224, 224, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.485f32, 0.456, 0.406], &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.229f32, 0.224, 0.225], &Device::Cpu)?.reshape((3, 1, 1))?;
    (data.to_dtype(candle_core::DType::F32)? / 255.0)?.broadcast_sub(&mean)?.broadcast_div(&std)
}
