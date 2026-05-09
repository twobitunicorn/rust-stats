//! Fitted-model results object.

pub struct OlsResults;

pub enum CovType {
    NonRobust,
    HC0,
    HC1,
    HC2,
    HC3,
}

pub struct Inference;
