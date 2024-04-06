//! MobileNet V2 implementation.
//!

use candle_core::{ Result, Module, Tensor, D };
use candle_nn::{ VarBuilder, Conv2d, BatchNorm, conv2d_no_bias, batch_norm, Linear, ops::dropout };

use crate::{ sequential::seq, sequential::Sequential };

/// Conv2D + BatchNorm2D + ReLU6
#[derive(Debug, Clone)]
pub struct Conv2dNormActivation {
    conv2d: Conv2d,
    batch_norm2d: BatchNorm,
}

impl Conv2dNormActivation {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize
    ) -> Result<Self> {
        let cfg = candle_nn::Conv2dConfig {
            stride,
            padding: (kernel_size - 1) / 2,
            groups,
            ..Default::default()
        };
        let conv2d = conv2d_no_bias(in_channels, out_channels, kernel_size, cfg, vb.pp(0))?;

        let batch_norm2d = batch_norm(out_channels, 1e-5, vb.pp(1))?;

        Ok(Self { conv2d, batch_norm2d })
    }
}

impl Module for Conv2dNormActivation {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        
        let ys = xs.apply(&self.conv2d)?;
        ys
        .apply_t(&self.batch_norm2d,false)?
        .relu()?
        .clamp(0.0, 6.0)
    }
}

#[derive(Debug,Clone)]
struct InvertedResidual{
    conv: ConvSequential
}

impl InvertedResidual {
    fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        expand_ratio: usize
    ) -> Result<Self> {
        Ok(
            Self { conv: ConvSequential::new(vb.pp("conv"), in_channels, out_channels, stride, expand_ratio)? }
        )
    }
}

impl Module for InvertedResidual{
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.conv)
    }
}

#[derive(Debug, Clone)]
struct ConvSequential {
    cbr1: Option<Conv2dNormActivation>,
    cbr2: Conv2dNormActivation,
    conv2d: Conv2d,
    batch_norm2d: BatchNorm,
    use_res_connect: bool,
}

impl ConvSequential {
    fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        expand_ratio: usize
    ) -> Result<Self> {
        let c_hidden = expand_ratio * in_channels;
        let mut id = 0;
        let cbr1 = if expand_ratio != 1 {
            // conv = conv.add(cbr(&p / id, c_in, c_hidden, 1, 1, 1));
            let cbr = Conv2dNormActivation::new(vb.pp(id), in_channels, c_hidden, 1, 1, 1)?;
            id += 1;
            Some(cbr)
        } else {
            None
        };
        let cbr2 = Conv2dNormActivation::new(vb.pp(id), c_hidden, c_hidden, 3, stride, c_hidden)?;
        let cfg = candle_nn::Conv2dConfig {
            stride: 1,
            ..Default::default()
        };
        let conv2d = conv2d_no_bias(c_hidden, out_channels, 1, cfg, vb.pp(id + 1))?;

        let batch_norm2d = batch_norm(out_channels, 1e-5, vb.pp(id + 2))?;
        let use_res_connect = stride == 1 && in_channels == out_channels;
        Ok(Self {
            cbr1,
            cbr2,
            conv2d,
            batch_norm2d,
            use_res_connect,
        })
    }
}

impl Module for ConvSequential {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let mut ys = xs.clone();
        if let Some(cbr1) = &self.cbr1 {
            ys = ys.apply(cbr1)?;
        }

        let ys = ys.apply(&self.cbr2)?.apply(&self.conv2d)?.apply_t(&self.batch_norm2d, false)?;

        if self.use_res_connect {
            xs + ys
        } else {
            Ok(ys)
        }
    }
}

const INVERTED_RESIDUAL_SETTINGS: [(usize, usize, usize, usize); 7] = [
    // (expand_ratio, c_out, n, stride)
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
];

#[derive(Debug, Clone)]
pub struct Features {
    cbr1: Conv2dNormActivation,
    invs: Sequential<InvertedResidual>,
    cbr2: Conv2dNormActivation,
}

impl Features {
    fn new(vb: VarBuilder) -> Result<Self> {
        let mut c_in = 32;
        let cbr1 = Conv2dNormActivation::new(vb.pp(0), 3, c_in, 3, 2, 1)?;
        let mut layer_id = 1;
        let mut invs = seq(0);
        for &(er, c_out, n, stride) in INVERTED_RESIDUAL_SETTINGS.iter() {
            for i in 0..n {
                let stride = if i == 0 { stride } else { 1 };
                let inv = InvertedResidual::new(vb.pp(layer_id), c_in, c_out, stride, er)?;
                invs.add(inv);
                c_in = c_out;
                layer_id += 1;
            }
        }
        let cbr2 = Conv2dNormActivation::new(vb.pp(layer_id),  c_in,1280, 1, 1, 1)?;

        Ok(Self {
            cbr1,
            invs,
            cbr2,
        })
    }
}

impl Module for Features {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let ys = xs.apply(&self.cbr1)?;
        
        let ys = ys.apply(&self.invs)?;
        ys.apply(&self.cbr2)
    }
}

#[derive(Debug, Clone)]
struct Classifier {
    linear: Linear,
}

impl Classifier {
    fn new(vb: VarBuilder, nclasses: usize) -> Result<Self> {
        let linear = candle_nn::linear(1280, nclasses, vb.pp(1))?;
        Ok(Self { linear })
    }
}

impl Module for Classifier {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let ys = dropout(xs, 0.2)?;
        ys.apply(&self.linear)
    }
}

#[derive(Debug, Clone)]
pub struct Mobilenetv2 {
    features: Features,
    classifier: Classifier,
}

impl Mobilenetv2 {
    pub fn new(vb: VarBuilder, nclasses: usize) -> Result<Self> {
        let features = Features::new(vb.pp("features"))?;
        let classifier = Classifier::new(vb.pp("classifier"), nclasses)?;
        Ok(Self { features, classifier })
    }
}

impl Module for Mobilenetv2 {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.features)?.mean(D::Minus1)?.mean(D::Minus1)?.apply(&self.classifier)
    }
}
