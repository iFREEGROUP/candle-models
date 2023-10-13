use candle_core::{ Result, D };
use candle_nn as nn;
use nn::{ Module, VarBuilder, Conv2d, Linear, BatchNormConfig };

/// 1x1 convolution
fn conv2d(
    vb: VarBuilder,
    in_planes: usize,
    out_planes: usize,
    ksize: usize,
    padding: usize,
    stride: usize,
    bias: bool
) -> Result<nn::Conv2d> {
    let conv_config = nn::Conv2dConfig {
        stride,
        padding,
        ..Default::default()
    };
    if bias {
        nn::conv2d(in_planes, out_planes, ksize, conv_config, vb)
    } else {
        nn::conv2d_no_bias(in_planes, out_planes, ksize, conv_config, vb)
    }
}

#[derive(Debug)]
pub struct PatchMerging {
    conv2d: nn::Conv2d,
    bn2: nn::BatchNorm,
}

impl PatchMerging {
    fn new(in_planes: usize, out_planes: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let conv2d = conv2d(vb.pp("0"), in_planes, out_planes, 1, 0, stride,false)?;

        let config = BatchNormConfig { eps: 1e-5, remove_mean: false, affine: true };
        let bn2 = nn::batch_norm(out_planes, config, vb.pp("1"))?;
        Ok(Self { conv2d, bn2 })
    }
}

impl Module for PatchMerging {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let xs = self.conv2d.forward(xs)?;
        let xs = self.bn2.forward(&xs)?;
        Ok(xs)
    }
}

fn downsample(
    vb: VarBuilder,
    in_planes: usize,
    out_planes: usize,
    stride: usize
) -> Result<Option<PatchMerging>> {
    if stride != 1 || in_planes != out_planes {
        Ok(Some(PatchMerging::new(in_planes, out_planes, stride, vb)?))
    } else {
        Ok(None)
    }
}

#[derive(Debug)]
pub struct BasicBlock {
    conv1: nn::Conv2d,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2d,
    bn2: nn::BatchNorm,
    downsample: Option<PatchMerging>,
}

impl BasicBlock {
    // (0): BasicBlock(
    //     (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    //     (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    //     (relu): ReLU(inplace=True)
    //     (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    //     (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    //     (downsample): Sequential(
    //       (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
    //       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    //     )
    //   )
    //   (1): BasicBlock(
    //     (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    //     (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    //     (relu): ReLU(inplace=True)
    //     (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    //     (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    //   )
    pub fn new(vb: VarBuilder, in_planes: usize, out_planes: usize, stride: usize) -> Result<Self> {
        let conv1 = conv2d(vb.pp("conv1"), in_planes, out_planes, 3, 1, stride,false)?;
        let bn1 = nn::batch_norm(out_planes, BatchNormConfig { eps: 1e-5, remove_mean: false, affine: true }, vb.pp("bn1"))?;
        let conv2 = conv2d(vb.pp("conv2"), out_planes, out_planes, 3, 1, 1,false)?;
        let bn2 = nn::batch_norm(out_planes, BatchNormConfig { eps: 1e-5, remove_mean: false, affine: true }, vb.pp("bn2"))?;
        let downsample = downsample(vb.pp("downsample"), in_planes, out_planes, stride)?;

        Ok(Self { conv1, bn1, conv2, bn2, downsample })
    }
}

impl Module for BasicBlock {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let ys = xs
            .apply(&self.conv1)?
            .apply(&self.bn1)?
            .relu()?
            .apply(&self.conv2)?
            .apply(&self.bn2)?;
        if let Some(downsample) = &self.downsample {
            (xs.apply(downsample) + ys)?.relu()
        } else {
            Ok(ys)
        }
    }
}

#[derive(Debug)]
pub struct BasicLayer{
    blocks:Vec<BasicBlock>
}

impl BasicLayer {
    pub fn new(
        vb: VarBuilder,
        in_planes: usize,
        out_planes: usize,
        stride: usize,
        cnt: usize
    ) -> Result<Self> {
        let mut layers = vec![];
        let layer = BasicBlock::new(vb.pp("0"), in_planes, out_planes, stride)?;
        layers.push(layer);
        for block_index in 1..cnt {
            let layer = BasicBlock::new(vb.pp(block_index.to_string()), out_planes, out_planes, 1)?;
            layers.push(layer);
        }

        Ok(BasicLayer{blocks: layers})
    }
}

impl Module for BasicLayer {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let mut ys = xs.apply(&self.blocks[0])?;
        for layer in &self.blocks[1..] {
            ys = ys.apply(layer)?;
        }
        Ok(ys)
    }
}

fn basic_layer(
    vb: VarBuilder,
    c_in: usize,
    c_out: usize,
    stride: usize,
    cnt: usize
) -> Result<BasicLayer> {
    BasicLayer::new(vb, c_in, c_out, stride, cnt)
}

#[derive(Debug)]
pub struct ResNet {
    conv1: Conv2d,
    bn1: nn::BatchNorm,
    layer1: BasicLayer,
    layer2: BasicLayer,
    layer3: BasicLayer,
    layer4: BasicLayer,
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
        let conv1 = conv2d(vb.pp("conv1"), 3, 64, 7, 3, 2,false)?;
        let bn1 = nn::batch_norm(
            64,
            BatchNormConfig { eps: 1e-5, remove_mean: false, affine: true },
            vb.pp("bn1")
        )?;
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
        let xs = xs.apply(&self.bn1)?;
        let xs = xs.relu()?;
        //nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        //
        let xs = xs.max_pool2d_with_stride(3, 2)?;
        println!("{xs}");
        let xs=xs.apply(&self.layer1)?;
        let xs=xs.apply(&self.layer2)?;
        let xs=xs.apply(&self.layer3)?;
        let xs = xs.apply(&self.layer4)?;
        // equivalent to adaptive_avg_pool2d([1, 1]) //avgpool
        let xs = xs.mean_keepdim(D::Minus2)?.mean_keepdim(D::Minus1)?; //[1, 512, 1, 1]
        let xs = xs.flatten_from(1)?; //[1,512]
        
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
///
/// Pre-trained weights can be downloaded at the following link:
/// <https://github.com/LaurentMazare/tch-rs/releases/download/untagged-eb220e5c19f9bb250bd1/resnet18.ot>
pub fn resnet18(vb: VarBuilder, num_classes: usize) -> Result<ResNet> {
    resnet(vb, Some(num_classes), 2, 2, 2, 2)
}

pub fn resnet18_no_final_layer(vb: VarBuilder) -> Result<ResNet> {
    resnet(vb, None, 2, 2, 2, 2)
}

/// Creates a ResNet-34 model.
///
/// Pre-trained weights can be downloaded at the following link:
/// <https://github.com/LaurentMazare/tch-rs/releases/download/untagged-eb220e5c19f9bb250bd1/resnet34.ot>
pub fn resnet34(vb: VarBuilder, num_classes: usize) -> Result<ResNet> {
    resnet(vb, Some(num_classes), 3, 4, 6, 3)
}

pub fn resnet34_no_final_layer(vb: VarBuilder) -> Result<ResNet> {
    resnet(vb, None, 3, 4, 6, 3)
}

// Bottleneck versions for ResNet 50, 101, and 152.
#[derive(Debug)]
pub struct BottleneckBlock {
    conv1: Conv2d,
    bn1: nn::BatchNorm,
    conv2: Conv2d,
    bn2: nn::BatchNorm,
    conv3: Conv2d,
    bn3: nn::BatchNorm,
    downsample: Option<PatchMerging>,
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
        let conv1 = conv2d(vb.pp("conv1"), in_planes, out_planes, 1, 0, 1,false)?;
        let bn1 = nn::batch_norm(
            out_planes,
            BatchNormConfig { eps: 1e-5, remove_mean: false, affine: true },
            vb.pp("bn1")
        )?;
        let conv2 = conv2d(vb.pp("conv2"), out_planes, out_planes, 3, 1, stride,false)?;
        let bn2 = nn::batch_norm(
            out_planes,
            BatchNormConfig { eps: 1e-5, remove_mean: false, affine: true },
            vb.pp("bn2")
        )?;

        let conv3 = conv2d(vb.pp("conv3"), out_planes, out_planes, 1, 0, 1,false)?;
        let bn3 = nn::batch_norm(
            out_planes,
            BatchNormConfig { eps: 1e-5, remove_mean: false, affine: true },
            vb.pp("bn3")
        )?;
        let downsample = downsample(vb.pp("downsample"), in_planes, e_dim, stride)?;
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
            .apply(&self.bn1)?
            .relu()?
            .apply(&self.conv2)?
            .apply(&self.bn2)?
            .relu()?
            .apply(&self.conv3)?
            .apply(&self.bn3)?;

        if let Some(downsample) = &self.downsample {
            (xs.apply(downsample) + ys)?.relu()
        } else {
            ys.relu()
        }
    }
}

#[derive(Debug)]
pub struct BottleneckLayer(pub Vec<BottleneckBlock>);

impl BottleneckLayer {
    pub fn new(
        vb: VarBuilder,
        in_planes: usize,
        out_planes: usize,
        stride: usize,
        cnt: usize
    ) -> Result<Self> {
        let mut blocks = vec![BottleneckBlock::new(vb.pp("0"), in_planes, out_planes, stride, 4)?];
        for block_index in 1..cnt {
            blocks.push(
                BottleneckBlock::new(
                    vb.pp(block_index.to_string()),
                    4 * out_planes,
                    out_planes,
                    1,
                    4
                )?
            );
        }

        Ok(BottleneckLayer(blocks))
    }
}

impl Module for BottleneckLayer {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let mut ys = xs.apply(&self.0[0])?;
        for block in &self.0[1..] {
            ys = ys.apply(block)?;
        }
        Ok(ys)
    }
}

fn bottleneck_layer(
    vb: VarBuilder,
    c_in: usize,
    c_out: usize,
    stride: usize,
    cnt: usize
) -> Result<BottleneckLayer> {
    BottleneckLayer::new(vb, c_in, c_out, stride, cnt)
}

#[derive(Debug)]
pub struct BottleneckResnet {
    conv1: Conv2d,
    bn1: nn::BatchNorm,
    layer1: BottleneckLayer,
    layer2: BottleneckLayer,
    layer3: BottleneckLayer,
    layer4: BottleneckLayer,
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
        let conv1 = conv2d(vb.pp("conv1"), 3, 64, 7, 3, 2,false)?;
        let bn1 = nn::batch_norm(
            64,
            BatchNormConfig { eps: 1e-5, remove_mean: false, affine: true },
            vb.pp("bn1")
        )?;
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
        let xs = xs
            .apply(&self.conv1)?
            .apply(&self.bn1)?
            .relu()?
            .max_pool2d_with_stride(3, 2)?
            .apply(&self.layer1)?
            .apply(&self.layer2)?
            .apply(&self.layer3)?
            .apply(&self.layer4)?;
        // equivalent to adaptive_avg_pool2d([1, 1])
        let xs = xs.mean_keepdim(D::Minus2)?.mean_keepdim(D::Minus1)?;
        let xs = xs.flatten_to(1)?;
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
