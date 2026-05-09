//! Fitted-model results object.

#[derive(Debug)]
pub struct OlsResults;

pub enum CovType {
    NonRobust,
    HC0,
    HC1,
    HC2,
    HC3,
}

pub struct Inference;
