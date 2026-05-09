use rust_stats::OlsError;

#[test]
fn error_variants_display_correctly() {
    let cases: Vec<(OlsError, &str)> = vec![
        (
            OlsError::DimensionMismatch { y: 10, x: 8 },
            "dimension mismatch: y has 10 rows but X has 8",
        ),
        (
            OlsError::InsufficientObservations { n: 3, p: 5 },
            "not enough observations: n=3 must exceed p=5",
        ),
        (
            OlsError::RankDeficient { rank: 2, p: 3 },
            "rank deficient design matrix: rank 2 < p 3",
        ),
        (OlsError::NonFinite, "input contains non-finite values"),
        (
            OlsError::NewXShapeMismatch { got: 4, expected: 3 },
            "predict X has 4 columns, expected 3",
        ),
        (
            OlsError::InvalidAlpha(1.5),
            "invalid alpha 1.5: must be in (0, 1)",
        ),
    ];
    for (err, expected) in cases {
        assert_eq!(format!("{}", err), expected);
    }
}
