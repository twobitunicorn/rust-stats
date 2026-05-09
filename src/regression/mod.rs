//! Regression models. v1: OLS only.

pub struct Ols<'a> {
    _phantom: core::marker::PhantomData<&'a ()>,
}

pub struct OlsResults;

pub enum CovType {
    NonRobust,
    HC0,
    HC1,
    HC2,
    HC3,
}

pub struct Inference;
