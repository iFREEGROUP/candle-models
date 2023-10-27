use candle_core::{ Result, Module, Tensor, D };
use candle_nn::{ Conv2d, VarBuilder, conv2d, Conv2dConfig, ops::dropout };

//max_pool2d([3, 3], [2, 2], [0, 0], [1, 1], true)
fn max_pool2d(xs: Tensor) -> Result<Tensor> {
    let xs = xs.pad_with_same(D::Minus1, 0, 0)?;
    let xs = xs.pad_with_same(D::Minus2, 0, 0)?;
    xs.max_pool2d_with_stride(3, 2)
}

#[derive(Debug, Clone)]
pub struct Fire {
    squeeze: Conv2d,
    expand1x1: Conv2d,
    expand3x3: Conv2d,
}

impl Fire {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        squeeze_planes: usize,
        expand1x1_planes: usize,
        expand3x3_planes: usize
    ) -> Result<Self> {
        let squeeze = conv2d(
            in_channels,
            squeeze_planes,
            1,
            Conv2dConfig { stride: 1, ..Default::default() },
            vb.pp("squeeze")
        )?;
        let expand1x1 = conv2d(
            squeeze_planes,
            expand1x1_planes,
            1,
            Conv2dConfig { stride: 1, ..Default::default() },
            vb.pp("expand1x1")
        )?;

        let expand3x3 = conv2d(
            squeeze_planes,
            expand3x3_planes,
            3,
            Conv2dConfig {
                stride: 1,
                padding: 1,
                ..Default::default()
            },
            vb.pp("expand3x3")
        )?;

        Ok(Self { squeeze, expand1x1, expand3x3 })
    }
}

impl Module for Fire {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let xs = xs.apply(&self.squeeze)?.relu()?;
        Tensor::cat(&[xs.apply(&self.expand1x1)?.relu()?, xs.apply(&self.expand3x3)?.relu()?], 1)
    }
}


#[derive(Debug, Clone)]
pub struct Classifier {
    conv2d: Conv2d,
}

impl Classifier {
    fn new(vb: VarBuilder, nclasses: usize) -> Result<Self> {
        let cfg = Conv2dConfig { stride: 1, ..Default::default() };
        let conv2d = conv2d(512, nclasses, 1, cfg, vb.pp("1"))?;

        Ok(Self { conv2d })
    }
}

impl Module for Classifier {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = dropout(xs, 0.5)?;
        let ys = ys.apply(&self.conv2d)?.relu()?;
        let ys = ys.mean(D::Minus1)?;
        ys.mean(D::Minus1)
    }
}

#[derive(Debug, Clone)]
pub enum Version {
    V1_0 {
        conv2d: Conv2d,
        fire3: Fire,
        fire4: Fire,
        fire5: Fire,
        fire7: Fire,
        fire8: Fire,
        fire9: Fire,
        fire10: Fire,
        fire12: Fire,
    },
    V1_1 {
        conv2d: Conv2d,
        fire3: Fire,
        fire4: Fire,
        fire6: Fire,
        fire7: Fire,
        fire9: Fire,
        fire10: Fire,
        fire11: Fire,
        fire12: Fire,
    },
}

impl Version {
    pub fn v1_0(vb: VarBuilder) -> Result<Self> {
        let initial_conv_cfg = Conv2dConfig { stride: 2, ..Default::default() };
        let conv2d = conv2d(3, 96, 7, initial_conv_cfg, vb.pp("0"))?;
        let fire3 = Fire::new(vb.pp("3"), 96, 16, 64, 64)?;
        let fire4 = Fire::new(vb.pp("4"), 128, 16, 64, 64)?;
        let fire5 = Fire::new(vb.pp("5"), 128, 32, 128, 128)?;
        let fire7 = Fire::new(vb.pp("7"), 256, 32, 128, 128)?;
        let fire8 = Fire::new(vb.pp("8"), 256, 48, 192, 192)?;
        let fire9 = Fire::new(vb.pp("9"), 384, 48, 192, 192)?;
        let fire10 = Fire::new(vb.pp("10"), 384, 64, 256, 256)?;
        let fire12 = Fire::new(vb.pp("12"), 512, 64, 256, 256)?;
        Ok(Self::V1_0 {
            conv2d,
            fire3,
            fire4,
            fire5,
            fire7,
            fire8,
            fire9,
            fire10,
            fire12,
        })
    }

    pub fn v1_1(vb: VarBuilder) -> Result<Self> {
        let initial_conv_cfg = Conv2dConfig { stride: 2, ..Default::default() };
        let conv2d = conv2d(3, 64, 3, initial_conv_cfg, vb.pp("0"))?;
        let fire3 = Fire::new(vb.pp("3"), 64, 16, 64, 64)?;
        let fire4 = Fire::new(vb.pp("4"), 128, 16, 64, 64)?;
        let fire6 = Fire::new(vb.pp("6"), 128, 32, 128, 128)?;
        let fire7 = Fire::new(vb.pp("7"), 256, 32, 128, 128)?;
        let fire9 = Fire::new(vb.pp("9"), 256, 48, 192, 192)?;
        let fire10 = Fire::new(vb.pp("10"), 384, 48, 192, 192)?;
        let fire11 = Fire::new(vb.pp("11"), 384, 64, 256, 256)?;
        let fire12 = Fire::new(vb.pp("12"), 512, 64, 256, 256)?;
        Ok(Self::V1_1 { conv2d, fire3, fire4, fire6, fire7, fire9, fire10, fire11, fire12 })
    }
}

impl Module for Version {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match &self {
            Version::V1_0 { conv2d, fire3, fire4, fire5, fire7, fire8, fire9, fire10, fire12 } => {
                let ys = xs.apply(conv2d)?.relu()?;
                let ys = max_pool2d(ys)?;
                let ys = ys.apply(fire3)?.apply(fire4)?.apply(fire5)?;
                let ys = max_pool2d(ys)?;
                let ys = ys.apply(fire7)?.apply(fire8)?.apply(fire9)?.apply(fire10)?;
                let ys = max_pool2d(ys)?;
                ys.apply(fire12)
            }
            Version::V1_1 { conv2d, fire3, fire4, fire6, fire7, fire9, fire10, fire11, fire12 } => {
                let ys = xs.apply(conv2d)?.relu()?;
                let ys = max_pool2d(ys)?;
                let ys = ys.apply(fire3)?.apply(fire4)?;
                let ys = max_pool2d(ys)?;
                let ys = ys.apply(fire6)?.apply(fire7)?;
                let ys = max_pool2d(ys)?;
                ys.apply(fire9)?.apply(fire10)?.apply(fire11)?.apply(fire12)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Squeezenet {
    features: Version,
    classifier: Classifier,
}

impl Squeezenet {
    pub fn new(vb: VarBuilder, features: Version,nclasses:usize) -> Result<Self> {
        let classifier = Classifier::new(vb.pp("classifier"), nclasses)?;
        Ok(Self { features, classifier })
    }
}

pub fn squeezenet1_0(vb: VarBuilder,nclasses:usize) -> Result<Squeezenet> {
    let features = Version::v1_0(vb.pp("features"))?;
    Squeezenet::new(vb, features,nclasses)
}

pub fn squeezenet1_1(vb: VarBuilder,nclasses:usize) -> Result<Squeezenet> {
    let features = Version::v1_1(vb.pp("features"))?;
    Squeezenet::new(vb, features,nclasses)
}

impl Module for Squeezenet {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.features)?.apply(&self.classifier)
    }
}
