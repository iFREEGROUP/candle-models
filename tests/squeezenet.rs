use candle_core::{ DType, Module, D, IndexOp };
use candle_nn::VarBuilder;
use candle_resnet::squeezenet::{squeezenet1_0, squeezenet1_1};
use common::load_image224;

use crate::imagenet::CLASSES;
mod imagenet;
mod common;


#[test]
fn test_squeezenet_1_0() -> candle_core::Result<()> {
    let model_file = "./testdata/squeezenet1_0.safetensors";
    let device = candle_core::Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };

    let image = load_image224("./testdata/mouse.jpg")?;

    let model = squeezenet1_0(vb, 1000)?;
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
fn test_squeezenet_1_1() -> candle_core::Result<()> {
    let model_file = "./testdata/squeezenet1_1.safetensors";
    let device = candle_core::Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };

    let image = load_image224("./testdata/mouse.jpg")?;

    let model = squeezenet1_1(vb, 1000)?;
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

