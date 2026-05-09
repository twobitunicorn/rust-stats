//! Cleveland 1990 STL (LOESS-based seasonal-trend decomposition).
//!
//! See module-level docs in `tsa::seasonal` for usage notes.

use crate::error::StlError;
use crate::tsa::seasonal::{Decomposition, StlOpts};
use faer::ColRef;

/// Cleveland 1990 STL — full implementation lands in Task 4.
pub fn stl(_y: ColRef<'_, f64>, _opts: StlOpts) -> Result<Decomposition, StlError> {
    unimplemented!("stl: implemented in Task 4")
}
