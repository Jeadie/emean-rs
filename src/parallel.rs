//! Loops that run on all cores when the `parallel` feature is on.
//!
//! Every helper gives each closure call a fixed piece of work that does not
//! depend on the number of threads, so results are bit-identical with and
//! without the feature and for any thread count.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Calls `f(index, chunk_a, chunk_b)` for matching chunks of `chunk` elements
/// of `a` and `b`. Each worker gets one scratch value from `init`.
pub(crate) fn for_each_chunk_pair<A, B, W, I, F>(
    a: &mut [A],
    b: &mut [B],
    chunk: usize,
    init: I,
    f: F,
) where
    A: Send,
    B: Send,
    I: Fn() -> W + Sync + Send,
    F: Fn(&mut W, usize, &mut [A], &mut [B]) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    a.par_chunks_mut(chunk)
        .zip(b.par_chunks_mut(chunk))
        .enumerate()
        .for_each_init(init, |scratch, (index, (a, b))| f(scratch, index, a, b));

    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = init();
        for (index, (a, b)) in a.chunks_mut(chunk).zip(b.chunks_mut(chunk)).enumerate() {
            f(&mut scratch, index, a, b);
        }
    }
}

/// Calls `f(index, row)` for each row of `width` elements of `data`.
pub(crate) fn for_each_row_mut<A, F>(data: &mut [A], width: usize, f: F)
where
    A: Send,
    F: Fn(usize, &mut [A]) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    data.par_chunks_mut(width)
        .enumerate()
        .for_each(|(index, row)| f(index, row));

    #[cfg(not(feature = "parallel"))]
    for (index, row) in data.chunks_mut(width).enumerate() {
        f(index, row);
    }
}

/// Calls `f(index, row)` for each row of `width` elements of `data`, and
/// returns the error of the lowest-indexed row that fails.
pub(crate) fn try_for_each_row<A, E, F>(data: &[A], width: usize, f: F) -> Result<(), E>
where
    A: Sync,
    E: Send,
    F: Fn(usize, &[A]) -> Result<(), E> + Sync + Send,
{
    #[cfg(feature = "parallel")]
    let first_error = data
        .par_chunks(width)
        .enumerate()
        .find_map_first(|(index, row)| f(index, row).err());

    #[cfg(not(feature = "parallel"))]
    let first_error = data
        .chunks(width)
        .enumerate()
        .find_map(|(index, row)| f(index, row).err());

    first_error.map_or(Ok(()), Err)
}

/// Whether loops run in parallel.
pub(crate) const fn enabled() -> bool {
    cfg!(feature = "parallel")
}
