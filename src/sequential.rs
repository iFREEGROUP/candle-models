use candle_core::{ Module, Tensor, Result };

#[derive(Debug, Clone)]
pub struct Sequential<T: Module> {
    layers: Vec<T>,
}

pub fn seq<T: Module>(cnt: usize) -> Sequential<T> {
    let v = if cnt == 0 { vec![] } else { Vec::with_capacity(cnt) };
    Sequential { layers: v }
}

impl<T: Module> Sequential<T> {
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn push(&mut self, layer: T) {
        self.layers.push(layer);
    }

    pub fn add(&mut self, layer: T) {
        self.layers.push(layer);
    }
}
impl<T: Module> Module for Sequential<T> {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = xs.apply(layer)?;
        }
        Ok(xs)
    }
}
