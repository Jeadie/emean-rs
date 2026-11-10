//! Small numerical building blocks.

/// Scales `values` to a Euclidean norm of 1. A zero vector stays zero.
///
/// The vector is first divided by its largest absolute value, so the sum of
/// squares can neither overflow nor underflow.
pub(crate) fn normalize_in_place(values: &mut [f64]) {
    let largest = values
        .iter()
        .fold(0.0f64, |acc, value| acc.max(value.abs()));
    if largest == 0.0 {
        return;
    }
    for value in values.iter_mut() {
        *value /= largest;
    }
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    for value in values.iter_mut() {
        *value /= norm;
    }
}

/// The median of each column of a row-major matrix with `dim` columns. For
/// an even number of rows, the lower of the two middle values.
pub(crate) fn column_medians(values: &[f64], dim: usize) -> Vec<f64> {
    let rows = values.len() / dim;
    if rows == 0 {
        return vec![0.0; dim];
    }
    let mut column = vec![0.0; rows];
    (0..dim)
        .map(|t| {
            for (out, row) in column.iter_mut().zip(values.chunks_exact(dim)) {
                *out = row[t];
            }
            *column
                .select_nth_unstable_by((rows - 1) / 2, f64::total_cmp)
                .1
        })
        .collect()
}

/// A running sum with Neumaier's compensation.
///
/// The error of the total does not grow with the number of terms, so the
/// objective of a pass over millions of rows keeps close to full `f64`
/// precision. Terms are added in order, so equal sequences give bit-identical
/// totals.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CompensatedSum {
    sum: f64,
    compensation: f64,
}

impl CompensatedSum {
    /// Adds one term.
    pub(crate) fn add(&mut self, term: f64) {
        let total = self.sum + term;
        self.compensation += if self.sum.abs() >= term.abs() {
            (self.sum - total) + term
        } else {
            (term - total) + self.sum
        };
        self.sum = total;
    }

    /// The compensated total.
    pub(crate) fn total(&self) -> f64 {
        self.sum + self.compensation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compensated_sum_is_exact_where_a_plain_sum_is_not() {
        let mut sum = CompensatedSum::default();
        for term in [1.0, 1e100, 1.0, -1e100] {
            sum.add(term);
        }
        assert_eq!(sum.total(), 2.0);
    }

    #[test]
    fn column_medians_ignore_outliers() {
        let values = [1.0, 10.0, 3.0, 30.0, 2.0, 1e30, 1e30, 20.0];
        assert_eq!(column_medians(&values, 2), [2.0, 20.0]);
        assert_eq!(column_medians(&[], 3), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn normalize_keeps_zero_and_scales_others() {
        let mut zero = [0.0, -0.0];
        normalize_in_place(&mut zero);
        assert_eq!(zero, [0.0, 0.0]);
        let mut v = [0.0, -3.0, 4.0];
        normalize_in_place(&mut v);
        assert_eq!(v, [0.0, -0.6, 0.8]);
        let mut tiny = [5e-324, 0.0];
        normalize_in_place(&mut tiny);
        assert_eq!(tiny, [1.0, 0.0]);
    }
}
