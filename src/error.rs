//! The error type of this crate.

use core::fmt;

/// An error from configuring, training or using an e-means model.
///
/// Every fallible operation checks its inputs before it changes any state,
/// so after an error the [`Session`](crate::Session) or model that returned
/// it is unchanged and can still be used. ([`Session::finish`](crate::Session::finish)
/// gives the session back inside [`Unfinished`](crate::Unfinished).)
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// A matrix was given a dimension of zero.
    ZeroDimension,
    /// A slice cannot be split into rows: its length is not a multiple of
    /// the dimension.
    RaggedRows {
        /// Length of the slice.
        len: usize,
        /// Requested row length.
        dim: usize,
    },
    /// A matrix has a different dimension from the one the model uses.
    DimensionMismatch {
        /// Dimension of the model.
        expected: usize,
        /// Dimension of the matrix that was given.
        actual: usize,
    },
    /// The initial centroids do not have one row per cluster.
    CentroidCountMismatch {
        /// Number of clusters.
        expected: u32,
        /// Number of rows that were given.
        actual: usize,
    },
    /// A set of centroids has no rows.
    NoCentroids,
    /// A set of centroids has more rows than a `u32` label can name.
    TooManyCentroids {
        /// Number of rows that were given.
        rows: usize,
    },
    /// There are fewer training rows than clusters.
    TooFewRows {
        /// Number of training rows.
        rows: usize,
        /// Number of clusters.
        clusters: u32,
    },
    /// A value is NaN or infinite.
    NonFinite {
        /// Row of the value, counted from the start of the matrix that was given.
        row: usize,
        /// Column of the value.
        column: usize,
    },
    /// A row is so long that squared distances to it can overflow the
    /// element type. The limit is a squared norm of `MAX / 16`, where `MAX`
    /// is the largest finite value of the element type (about `8.5e18` for
    /// the norm of an `f32` row).
    MagnitudeTooLarge {
        /// Row of the vector, counted from the start of the matrix that was given.
        row: usize,
    },
    /// Spherical clustering got a row whose Euclidean norm is not 1 within
    /// [`UNIT_NORM_TOLERANCE`](crate::UNIT_NORM_TOLERANCE).
    /// [`normalize_rows`](crate::normalize_rows) fixes this.
    NotUnitNorm {
        /// Row of the vector, counted from the start of the matrix that was given.
        row: usize,
        /// Euclidean norm of the row.
        norm: f64,
    },
    /// An epoch budget is not a positive, finite number of at most
    /// [`Epochs::MAX`](crate::Epochs::MAX).
    InvalidEpochs {
        /// The rejected value.
        value: f64,
    },
    /// A step size is not in the interval (0, 1].
    InvalidStep {
        /// The rejected value.
        value: f64,
    },
    /// A revival threshold is not a positive, finite number.
    InvalidThreshold {
        /// The rejected value.
        value: f64,
    },
    /// A fractional epoch budget below one rounds to zero rows, so there is
    /// nothing to train on.
    EmptyBudget {
        /// The epoch budget.
        epochs: f64,
        /// Number of training rows.
        rows: usize,
    },
    /// The number of rows or updates in a run does not fit in a `u64`.
    BudgetTooLarge,
    /// A [`Session`](crate::Session) was fed more rows than its recipe
    /// contains.
    BudgetExceeded {
        /// Rows that the recipe still expects.
        remaining: u64,
        /// Rows that were given.
        offered: usize,
    },
    /// [`Session::finish`](crate::Session::finish) was called before the
    /// recipe's rows were all fed. This is what
    /// [`Unfinished`](crate::Unfinished) converts to.
    Incomplete {
        /// Rows that the recipe still expects.
        remaining: u64,
    },
    /// An output buffer does not have one element per input row.
    BufferLength {
        /// Number of input rows.
        expected: usize,
        /// Length of the buffer.
        actual: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::ZeroDimension => write!(f, "the dimension must be at least 1"),
            Self::RaggedRows { len, dim } => write!(
                f,
                "a slice of {len} elements cannot be split into rows of {dim} elements"
            ),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "expected vectors of dimension {expected}, got dimension {actual}"
            ),
            Self::CentroidCountMismatch { expected, actual } => write!(
                f,
                "expected {expected} initial centroids (one per cluster), got {actual}"
            ),
            Self::NoCentroids => write!(f, "there must be at least one centroid"),
            Self::TooManyCentroids { rows } => {
                write!(f, "{rows} centroids are more than a u32 label can name")
            }
            Self::TooFewRows { rows, clusters } => write!(
                f,
                "cannot make {clusters} clusters from {rows} rows: \
                 there must be at least as many rows as clusters"
            ),
            Self::NonFinite { row, column } => {
                write!(
                    f,
                    "the value at row {row}, column {column} is NaN or infinite"
                )
            }
            Self::MagnitudeTooLarge { row } => write!(
                f,
                "row {row} is too long: its squared norm can overflow the element type"
            ),
            Self::NotUnitNorm { row, norm } => write!(
                f,
                "spherical clustering needs rows of norm 1, but row {row} has norm {norm}; \
                 use emeans::normalize_rows first"
            ),
            Self::InvalidEpochs { value } => write!(
                f,
                "the epoch budget must be positive, finite and at most {}, got {value}",
                crate::Epochs::MAX.get()
            ),
            Self::InvalidStep { value } => {
                write!(f, "the step size must be in (0, 1], got {value}")
            }
            Self::InvalidThreshold { value } => write!(
                f,
                "the revival threshold must be positive and finite, got {value}"
            ),
            Self::EmptyBudget { epochs, rows } => write!(
                f,
                "{epochs} epochs of {rows} rows round to zero rows: there is nothing to train on"
            ),
            Self::BudgetTooLarge => {
                write!(
                    f,
                    "the number of rows or updates in this run does not fit in a u64"
                )
            }
            Self::BudgetExceeded { remaining, offered } => write!(
                f,
                "got {offered} rows, but the recipe expects only {remaining} more"
            ),
            Self::Incomplete { remaining } => write!(
                f,
                "training is not complete: the recipe expects {remaining} more rows"
            ),
            Self::BufferLength { expected, actual } => write!(
                f,
                "the output buffer has {actual} elements, but there are {expected} rows"
            ),
        }
    }
}

impl std::error::Error for Error {}
