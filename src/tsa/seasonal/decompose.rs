//! Classical (moving-average) seasonal-trend decomposition.
//!
//! See module-level docs in `tsa::seasonal` for usage notes.

use crate::error::SeasonalDecomposeError;
use crate::tsa::seasonal::{Decomposition, SeasonalDecomposeOpts};
use faer::ColRef;

/// Classical decomposition — full implementation lands in Task 5.
pub fn seasonal_decompose(
    _y: ColRef<'_, f64>,
    _opts: SeasonalDecomposeOpts,
) -> Result<Decomposition, SeasonalDecomposeError> {
    unimplemented!("seasonal_decompose: implemented in Task 5")
}
