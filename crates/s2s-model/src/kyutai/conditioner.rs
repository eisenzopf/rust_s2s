//! Conditioners (`moshi::conditioner`): vectors added to every input embedding.
//!
//! - **LUT**: `output_proj(embed[value])` for a categorical value (Moshi's
//!   `description` quality tag).
//! - **Continuous attribute**: `output_proj(cat(cos(v s f), sin(v s f)))` with
//!   `f_i = 1 / max_period^(i / (dim/2 - 1))`, `s = scale_factor` (the DSM STT
//!   `delay` conditioner).
//! - **Learnt padding**: a stored vector per conditioner, used when the
//!   condition is absent.
//!
//! Tensor names: `condition_provider.conditioners.{name}.{embed.weight,
//! output_proj.weight, learnt_padding}`.

use std::collections::BTreeMap;

use super::config::ConditionerConfig;
use super::error::{Error, Result};
use super::layers::Linear;
use super::weights::{join, Loader};

/// A condition: a `[d_model]` vector added to the summed input embedding
/// (Kyutai's `Condition::AddToInput`).
#[derive(Clone, Debug, PartialEq)]
pub struct Condition(pub Vec<f32>);

#[derive(Clone, Debug)]
enum Conditioner {
    Lut {
        embed: Vec<f32>,
        dim: usize,
        output_proj: Linear,
        learnt_padding: Vec<f32>,
        values: BTreeMap<String, usize>,
    },
    Continuous {
        dim: usize,
        scale_factor: f32,
        max_period: f32,
        output_proj: Linear,
        learnt_padding: Vec<f32>,
    },
}

/// The model's conditioners, by name.
#[derive(Clone, Debug)]
pub struct ConditionProvider {
    conditioners: BTreeMap<String, Conditioner>,
}

impl ConditionProvider {
    pub(crate) fn load(
        l: &mut Loader,
        prefix: &str,
        d_model: usize,
        cfg: &BTreeMap<String, ConditionerConfig>,
    ) -> Result<Self> {
        let prefix = join(prefix, "conditioners");
        let mut conditioners = BTreeMap::new();
        for (name, c) in cfg {
            let p = join(&prefix, name);
            let c = match c {
                ConditionerConfig::Lut(c) => Conditioner::Lut {
                    embed: l.get(&join(&p, "embed.weight"), &[c.n_bins + 1, c.dim])?,
                    dim: c.dim,
                    output_proj: Linear::load(l, &join(&p, "output_proj"), c.dim, d_model, false)?,
                    learnt_padding: l.get(&join(&p, "learnt_padding"), &[1, 1, d_model])?,
                    values: c
                        .possible_values
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (v.clone(), i))
                        .collect(),
                },
                ConditionerConfig::ContinuousAttribute(c) => Conditioner::Continuous {
                    dim: c.dim,
                    scale_factor: c.scale_factor,
                    max_period: c.max_period,
                    output_proj: Linear::load(l, &join(&p, "output_proj"), c.dim, d_model, false)?,
                    learnt_padding: l.get(&join(&p, "learnt_padding"), &[1, 1, d_model])?,
                },
            };
            conditioners.insert(name.clone(), c);
        }
        Ok(Self { conditioners })
    }

    fn get(&self, name: &str) -> Result<&Conditioner> {
        self.conditioners
            .get(name)
            .ok_or_else(|| Error::Input(format!("unknown conditioner {name}")))
    }

    /// Condition on a categorical value of a LUT conditioner.
    pub fn condition_lut(&self, name: &str, value: &str) -> Result<Condition> {
        match self.get(name)? {
            Conditioner::Lut {
                embed,
                dim,
                output_proj,
                values,
                ..
            } => {
                let idx = *values.get(value).ok_or_else(|| {
                    Error::Input(format!("unknown value for lut conditioner '{value}'"))
                })?;
                Ok(Condition(
                    output_proj.forward(&embed[idx * dim..(idx + 1) * dim]),
                ))
            }
            _ => Err(Error::Input(format!(
                "cannot use conditioner with a str value {name}"
            ))),
        }
    }

    /// Condition on a scalar of a continuous conditioner.
    pub fn condition_cont(&self, name: &str, value: f32) -> Result<Condition> {
        match self.get(name)? {
            Conditioner::Continuous {
                dim,
                scale_factor,
                max_period,
                output_proj,
                ..
            } => {
                let half = dim / 2;
                let v = value * scale_factor;
                let mut emb = vec![0.0f32; 2 * half];
                for i in 0..half {
                    let f = v * (1f32 / max_period.powf(i as f32 / (half - 1) as f32));
                    emb[i] = f.cos();
                    emb[half + i] = f.sin();
                }
                Ok(Condition(output_proj.forward(&emb)))
            }
            _ => Err(Error::Input(format!(
                "cannot use conditioner with a float value {name}"
            ))),
        }
    }

    /// The learnt padding of a conditioner (used when its value is absent).
    pub fn learnt_padding(&self, name: &str) -> Result<Condition> {
        Ok(Condition(match self.get(name)? {
            Conditioner::Lut { learnt_padding, .. }
            | Conditioner::Continuous { learnt_padding, .. } => learnt_padding.clone(),
        }))
    }

    /// Conditioner names.
    pub fn names(&self) -> Vec<&str> {
        self.conditioners.keys().map(|s| s.as_str()).collect()
    }
}
