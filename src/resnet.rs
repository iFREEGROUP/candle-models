//! 参考了以下两个实现 tch (libtorch) 和 pytorch
//! https://github.com/LaurentMazare/tch-rs/blob/main/src/vision/resnet.rs
//! https://github.com/pytorch/vision/blob/0d75d9e5516f446c9c0ef93bd4ed9fea13992d06/torchvision/models/resnet.py

use candle_core::{ Result, D };
use candle_nn as nn;
use nn::{ Module, VarBuilder, Conv2d, Linear, batch_norm };

use crate::sequential::{Sequential, seq};

/// 1x1 convolution
fn conv2d(
    in_planes: usize,
    out_planes: usize,
    ksize: usize,
    padding: usize,
    stride: usize,
    vb: VarBuilder
) -> Result<Conv2d> {
    let conv2d_cfg = candle_nn::Conv2dConfig {
        stride,
        padding,
        ..Default::default()
    };
    candle_nn::conv2d_no_bias(in_planes, out_planes, ksize, conv2d_cfg, vb)
}

#[derive(Debug,Clone)]
pub struct Downsample {
    conv2d: nn::Conv2d,
    bn2: nn::BatchNorm,
    in_planes: usize,
    out_planes: usize,
    stride: usize,
}

impl Module for Downsample {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        if self.stride != 1 || self.in_planes != self.out_planes {
            xs.apply(&self.conv2d)?.apply_t(&self.bn2,false)
        } else {
            Ok(xs.clone())
        }
    }
}


fn downsample(in_planes: usize, out_planes: usize, stride: usize, vb: VarBuilder) -> Result<Option<Downsample>> {
    if stride != 1 || in_planes != out_planes {
        let conv = conv2d(in_planes, out_planes, 1, 0, stride, vb.pp(0))?;
        let bn = batch_norm(out_planes, 1e-5, vb.pp(1))?;
        Ok(
            Some(Downsample{ conv2d: conv, bn2: bn, in_planes, out_planes, stride})
        )
    } else {
        Ok(None)
    }
}


#[derive(Debug,Clone)]
pub struct BasicBlock {
    conv1: nn::Conv2d,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2d,
    bn2: nn::BatchNorm,
    downsample: Option<Downsample>,
}

impl BasicBlock {
    pub fn new(vb: VarBuilder, in_planes: usize, out_planes: usize, stride: usize) -> Result<Self> {
        let conv1 = conv2d(in_planes, out_planes, 3, 1, stride, vb.pp("conv1"))?;

        let bn1 = batch_norm(out_planes, 1e-5, vb.pp("bn1"))?;
        let conv2 = conv2d(out_planes, out_planes, 3, 1, 1, vb.pp("conv2"))?;
        let bn2 = batch_norm(out_planes, 1e-5, vb.pp("bn2"))?;
        let downsample = downsample(in_planes, out_planes, stride, vb.pp("downsample"))?;

        Ok(Self { conv1, bn1, conv2, bn2, downsample })
    }
}

impl Module for BasicBlock {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let ys = xs
            .apply(&self.conv1)?
            .apply_t(&self.bn1,false)?
            .relu()?
            .apply(&self.conv2)?
            .apply_t(&self.bn2,false)?;

        // (xs.apply(&self.downsample) + ys)?.relu()
        if let Some(downsample) = &self.downsample {
            (xs.apply(downsample) + ys)?.relu()
        } else {
            (ys + xs)?.relu()
        }
    }
}

fn basic_layer(
    vb: VarBuilder,
    in_planes: usize,
    out_planes: usize,
    stride: usize,
    cnt: usize
) -> Result<Sequential<BasicBlock>> {
    let mut layers = seq(cnt);
    for block_index in 0..cnt {
        let l_in = if block_index == 0 { in_planes } else { out_planes };
        let stride = if block_index == 0 { stride } else { 1 };
        // let layer = basic_block(l_in, out_planes, stride, vb.pp(block_index))?;
        let layer = BasicBlock::new(vb.pp(block_index.to_string()), l_in, out_planes, stride)?;
        layers.push(layer);
    }
    Ok(layers)
}

#[derive(Debug,Clone)]
pub struct ResNet {
    conv1: Conv2d,
    bn1: nn::BatchNorm,
    layer1: Sequential<BasicBlock>,
    layer2: Sequential<BasicBlock>,
    layer3: Sequential<BasicBlock>,
    layer4: Sequential<BasicBlock>,
    linear: Option<Linear>,
}

impl ResNet {
    pub fn new(
        vb: VarBuilder,
        nclasses: Option<usize>,
        c1: usize,
        c2: usize,
        c3: usize,
        c4: usize
    ) -> Result<Self> {
        let conv1 = conv2d(3, 64, 7, 3, 2, vb.pp("conv1"))?;
        let bn1 = batch_norm(64, 1e-5, vb.pp("bn1"))?;
        let layer1 = basic_layer(vb.pp("layer1"), 64, 64, 1, c1)?;
        let layer2 = basic_layer(vb.pp("layer2"), 64, 128, 2, c2)?;
        let layer3 = basic_layer(vb.pp("layer3"), 128, 256, 2, c3)?;
        let layer4 = basic_layer(vb.pp("layer4"), 256, 512, 2, c4)?;

        let linear = if let Some(n) = nclasses {
            Some(nn::linear(512, n, vb.pp("fc"))?)
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            layer1,
            layer2,
            layer3,
            layer4,
            linear,
        })
    }
}

impl Module for ResNet {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let xs = xs.apply(&self.conv1)?;
        let xs = xs.apply_t(&self.bn1,false)?;
        let xs = xs.relu()?;

        let xs = xs.pad_with_same(D::Minus1, 1, 1)?;
        let xs = xs.pad_with_same(D::Minus2, 1, 1)?;
        let xs = xs.max_pool2d_with_stride(3, 2)?;

        let xs = xs.apply(&self.layer1)?;
        let xs = xs.apply(&self.layer2)?;
        let xs = xs.apply(&self.layer3)?;
        let xs = xs.apply(&self.layer4)?;

        // Equivalent to adaptive_avg_pool2d([1, 1]) -> squeeze(-1) -> squeeze(-1)
        let xs = xs.mean(D::Minus1)?;
        let xs = xs.mean(D::Minus1)?;

        match &self.linear {
            Some(fc) => xs.apply(fc),
            None => Ok(xs),
        }
    }
}

fn resnet(
    vb: VarBuilder,
    nclasses: Option<usize>,
    c1: usize,
    c2: usize,
    c3: usize,
    c4: usize
) -> Result<ResNet> {
    ResNet::new(vb, nclasses, c1, c2, c3, c4)
}

/// Creates a ResNet-18 model.
pub fn resnet18(vb: VarBuilder, num_classes: usize) -> Result<ResNet> {
    resnet(vb, Some(num_classes), 2, 2, 2, 2)
}

pub fn resnet18_no_final_layer(vb: VarBuilder) -> Result<ResNet> {
    resnet(vb, None, 2, 2, 2, 2)
}

/// Creates a ResNet-34 model.
pub fn resnet34(vb: VarBuilder, num_classes: usize) -> Result<ResNet> {
    resnet(vb, Some(num_classes), 3, 4, 6, 3)
}

pub fn resnet34_no_final_layer(vb: VarBuilder) -> Result<ResNet> {
    resnet(vb, None, 3, 4, 6, 3)
}

// Bottleneck versions for ResNet 50, 101, and 152.
#[derive(Debug,Clone)]
pub struct BottleneckBlock {
    conv1: Conv2d,
    bn1: nn::BatchNorm,
    conv2: Conv2d,
    bn2: nn::BatchNorm,
    conv3: Conv2d,
    bn3: nn::BatchNorm,
    downsample: Option<Downsample>,
}

impl BottleneckBlock {
    pub fn new(
        vb: VarBuilder,
        in_planes: usize,
        out_planes: usize,
        stride: usize,
        e: usize
    ) -> Result<Self> {
        let e_dim = e * out_planes;
        let conv1 = conv2d(in_planes, out_planes, 1, 0, 1, vb.pp("conv1"))?;
        let bn1 = nn::batch_norm(out_planes, 1e-5, vb.pp("bn1"))?;
        let conv2 = conv2d(out_planes, out_planes, 3, 1, stride, vb.pp("conv2"))?;
        let bn2 = nn::batch_norm(out_planes, 1e-5, vb.pp("bn2"))?;

        let conv3 = conv2d(out_planes, e_dim, 1, 0, 1, vb.pp("conv3"))?;
        let bn3 = nn::batch_norm(e_dim, 1e-5, vb.pp("bn3"))?;
        let downsample = downsample(in_planes, e_dim, stride, vb.pp("downsample"))?;
        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            downsample,
        })
    }
}

impl Module for BottleneckBlock {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let ys = xs
            .apply(&self.conv1)?
            .apply_t(&self.bn1,false)?
            .relu()?
            .apply(&self.conv2)?
            .apply_t(&self.bn2,false)?
            .relu()?
            .apply(&self.conv3)?
            .apply_t(&self.bn3,false)?;

        if let Some(downsample) = &self.downsample {
            (xs.apply(downsample) + ys)?.relu()
        } else {
            (ys + xs)?.relu()
        }
    }
}

fn bottleneck_layer(
    vb: VarBuilder,
    in_planes: usize,
    out_planes: usize,
    stride: usize,
    cnt: usize
) -> Result<Sequential<BottleneckBlock>> {
    let mut layers = seq(cnt);
    for index in 0..cnt {
        let l_in = if index == 0 { in_planes } else { 4 * out_planes };
        let stride = if index == 0 { stride } else { 1 };
        layers.push(
            BottleneckBlock::new(vb.pp(index), l_in, out_planes, stride, 4)?
        );
    }
    Ok(layers)
    
}

#[derive(Debug,Clone)]
pub struct BottleneckResnet {
    conv1: Conv2d,
    bn1: nn::BatchNorm,
    layer1: Sequential<BottleneckBlock>,
    layer2: Sequential<BottleneckBlock>,
    layer3: Sequential<BottleneckBlock>,
    layer4: Sequential<BottleneckBlock>,
    linear: Option<Linear>,
}

impl BottleneckResnet {
    pub fn new(
        vb: VarBuilder,
        nclasses: Option<usize>,
        c1: usize,
        c2: usize,
        c3: usize,
        c4: usize
    ) -> Result<Self> {
        let conv1 = conv2d(3, 64, 7, 3, 2, vb.pp("conv1"))?;
        let bn1 = nn::batch_norm(64, 1e-5, vb.pp("bn1"))?;
        let layer1 = bottleneck_layer(vb.pp("layer1"), 64, 64, 1, c1)?;
        let layer2 = bottleneck_layer(vb.pp("layer2"), 4 * 64, 128, 2, c2)?;
        let layer3 = bottleneck_layer(vb.pp("layer3"), 4 * 128, 256, 2, c3)?;
        let layer4 = bottleneck_layer(vb.pp("layer4"), 4 * 256, 512, 2, c4)?;

        let linear = if let Some(n) = nclasses {
            Some(nn::linear(4 * 512, n, vb.pp("fc"))?)
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            layer1,
            layer2,
            layer3,
            layer4,
            linear,
        })
    }
}

impl Module for BottleneckResnet {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let xs = xs.apply(&self.conv1)?;
        let xs = xs.apply_t(&self.bn1,false)?;
        let xs = xs.relu()?;

        let xs = xs.pad_with_same(D::Minus1, 1, 1)?;
        let xs = xs.pad_with_same(D::Minus2, 1, 1)?;
        let xs = xs.max_pool2d_with_stride(3, 2)?;

        let xs = xs.apply(&self.layer1)?;
        let xs = xs.apply(&self.layer2)?;
        let xs = xs.apply(&self.layer3)?;
        let xs = xs.apply(&self.layer4)?;

        // Equivalent to adaptive_avg_pool2d([1, 1]) -> squeeze(-1) -> squeeze(-1)
        let xs = xs.mean(D::Minus1)?;
        let xs = xs.mean(D::Minus1)?;

        match &self.linear {
            Some(fc) => xs.apply(fc),
            None => Ok(xs),
        }
    }
}

fn bottleneck_resnet(
    vb: VarBuilder,
    nclasses: Option<usize>,
    c1: usize,
    c2: usize,
    c3: usize,
    c4: usize
) -> Result<BottleneckResnet> {
    BottleneckResnet::new(vb, nclasses, c1, c2, c3, c4)
}

pub fn resnet50(vb: VarBuilder, num_classes: usize) -> Result<BottleneckResnet> {
    bottleneck_resnet(vb, Some(num_classes), 3, 4, 6, 3)
}

pub fn resnet50_no_final_layer(vb: VarBuilder) -> Result<BottleneckResnet> {
    bottleneck_resnet(vb, None, 3, 4, 6, 3)
}

pub fn resnet101(vb: VarBuilder, num_classes: usize) -> Result<BottleneckResnet> {
    bottleneck_resnet(vb, Some(num_classes), 3, 4, 23, 3)
}

pub fn resnet101_no_final_layer(vb: VarBuilder) -> Result<BottleneckResnet> {
    bottleneck_resnet(vb, None, 3, 4, 23, 3)
}

pub fn resnet152(vb: VarBuilder, num_classes: usize) -> Result<BottleneckResnet> {
    bottleneck_resnet(vb, Some(num_classes), 3, 8, 36, 3)
}

pub fn resnet150_no_final_layer(vb: VarBuilder) -> Result<BottleneckResnet> {
    bottleneck_resnet(vb, None, 3, 8, 36, 3)
}
