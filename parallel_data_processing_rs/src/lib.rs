#![feature(portable_simd)]
use ndarray::s;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::iter::repeat;
use std::mem::size_of;
use std::slice;
use std::thread;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64;
use std::arch::x86_64::__m256d;

use std::simd::f64x4;

const CORES: usize = 6;

trait SimdLen {
    const SIMD_LEN: usize;
}

impl SimdLen for f64 {
    const SIMD_LEN: usize = 32 / size_of::<f64>();
}

#[pymodule]
fn parallel_data_processing_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(stdev, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    m.add_function(wrap_pyfunction!(sum_simd, m)?)?;
    m.add_function(wrap_pyfunction!(mean_simd, m)?)?;
    m.add_function(wrap_pyfunction!(var_simd, m)?)?;
    m.add_function(wrap_pyfunction!(stdev_simd, m)?)?;
    m.add_function(wrap_pyfunction!(max_simd, m)?)?;
    m.add_function(wrap_pyfunction!(min_simd, m)?)?;
    Ok(())
}

#[pyfunction]
fn sum(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(sum_arr(&arr))
}

#[pyfunction]
fn sum_simd(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(sum_arr_simd(&arr))
}

#[pyfunction]
fn mean(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(mean_arr(&arr))
}

#[pyfunction]
fn mean_simd(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(mean_arr_simd(&arr))
}

#[pyfunction]
fn var(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(var_arr(&arr))
}

#[pyfunction]
fn var_simd(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(var_arr_simd(&arr))
}

#[pyfunction]
fn stdev(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(stdev_arr(&arr))
}

#[pyfunction]
fn stdev_simd(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(stdev_arr_simd(&arr))
}

#[pyfunction]
fn max(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(max_arr(&arr))
}

#[pyfunction]
fn max_simd(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(max_arr_simd(&arr))
}

#[pyfunction]
fn min(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(min_arr(&arr))
}

#[pyfunction]
fn min_simd(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(min_arr_simd(&arr))
}

fn sum_arr(arr: &ndarray::Array1<f64>) -> f64 {
    if arr.len() < CORES {
        return arr.sum();
    }

    let indexes = indexing_reps(arr.len());
    let arr_ptr = arr.as_ptr() as usize;
    let mut handles = Vec::with_capacity(CORES);

    for (i, &rep) in indexes.iter().enumerate() {
        let handle = thread::spawn(move || {
            let local_ptr = unsafe { (arr_ptr as *const f64).add(i) };
            let mut sum = 0.0;
            for r in 0..rep {
                sum += unsafe { *local_ptr.add(CORES * r) }
            }
            sum
        });
        handles.push(handle);
    }

    let mut sum = 0.0;
    for handle in handles {
        sum += handle.join().expect("Error Joining a thread");
    }
    sum
}

fn sum_arr_simd(arr: &ndarray::Array1<f64>) -> f64 {
    if arr.len() < CORES * f64::SIMD_LEN {
        return arr.sum();
    }

    let (reps, leftover) = chunking_reps(&arr);
    let arr_ptr = arr.as_ptr() as usize;
    let mut handles = Vec::with_capacity(CORES);

    for (i, &rep) in reps.iter().enumerate() {
        let handle = thread::spawn(move || {
            let local_ptr = unsafe { (arr_ptr as *const f64).add(i * f64::SIMD_LEN) };
            let mut vsum = unsafe { x86_64::_mm256_setzero_pd() };

            for r in 0..rep {
                let chunk = unsafe {
                    slice::from_raw_parts(local_ptr.add(CORES * f64::SIMD_LEN * r), f64::SIMD_LEN)
                };
                let chunk = __m256d::from(f64x4::from_slice(chunk));
                vsum = unsafe { x86_64::_mm256_add_pd(vsum, chunk) };
            }

            let vsum_128l = unsafe { x86_64::_mm256_castpd256_pd128(vsum) };
            let vsum_128h = unsafe { x86_64::_mm256_extractf128_pd::<1>(vsum) };
            let vwsum = unsafe { x86_64::_mm_add_pd(vsum_128l, vsum_128h) };
            let vwsum_64h = unsafe { x86_64::_mm_unpackhi_pd(vwsum, vwsum) };
            unsafe { x86_64::_mm_cvtsd_f64(x86_64::_mm_add_sd(vwsum, vwsum_64h)) }
        });
        handles.push(handle);
    }

    let mut sum: f64 = arr.slice(s![arr.len() - leftover..]).iter().sum();
    for handle in handles {
        sum += handle.join().expect("Error Joining a thread");
    }

    sum
}

fn mean_arr(arr: &ndarray::Array1<f64>) -> f64 {
    sum_arr(&arr) / arr.len() as f64
}

fn mean_arr_simd(arr: &ndarray::Array1<f64>) -> f64 {
    sum_arr_simd(&arr) / arr.len() as f64
}

fn var_arr(arr: &ndarray::Array1<f64>) -> f64 {
    if arr.len() < CORES {
        return arr.var(0.0);
    }

    let mean = mean_arr(&arr);

    let indexes = indexing_reps(arr.len());
    let arr_ptr = arr.as_ptr() as usize;
    let mut handles = Vec::with_capacity(CORES);

    for (i, &rep) in indexes.iter().enumerate() {
        let local_ptr = arr_ptr.clone();
        let handle = thread::spawn(move || {
            let local_ptr = unsafe { (local_ptr as *const f64).add(i) };
            let mut sum = 0.0;
            for r in 0..rep {
                sum += (unsafe { *local_ptr.add(CORES * r) } - mean).powi(2);
            }
            sum
        });
        handles.push(handle);
    }

    let mut sum = 0.0;
    for handle in handles {
        sum += handle.join().expect("Error Joining a thread");
    }
    sum / arr.len() as f64
}

fn var_arr_simd(arr: &ndarray::Array1<f64>) -> f64 {
    if arr.len() < CORES * f64::SIMD_LEN {
        return arr.var(0.0);
    }
    let mean = mean_arr_simd(&arr);
    let (indexes, leftover) = chunking_reps(&arr);
    let arr_ptr = arr.as_ptr() as usize;
    let mut handles = Vec::with_capacity(CORES);
    for (i, &rep) in indexes.iter().enumerate() {
        let handle = thread::spawn(move || {
            let mut local_ptr = arr_ptr as *const f64;
            local_ptr = unsafe { local_ptr.add(i * f64::SIMD_LEN) };
            let local_mean = unsafe { x86_64::_mm256_set1_pd(mean) };
            let mut vsum = unsafe { x86_64::_mm256_setzero_pd() };
            for r in 0..rep {
                let chunk = unsafe {
                    slice::from_raw_parts(local_ptr.add(CORES * f64::SIMD_LEN * r), f64::SIMD_LEN)
                };
                let mut chunk = __m256d::from(f64x4::from_slice(chunk));
                chunk = unsafe { x86_64::_mm256_sub_pd(chunk, local_mean) };
                chunk = unsafe { x86_64::_mm256_mul_pd(chunk, chunk) };
                vsum = unsafe { x86_64::_mm256_add_pd(vsum, chunk) };
            }
            let vsum_128l = unsafe { x86_64::_mm256_castpd256_pd128(vsum) };
            let vsum_128h = unsafe { x86_64::_mm256_extractf128_pd::<1>(vsum) };
            let vwsum = unsafe { x86_64::_mm_add_pd(vsum_128l, vsum_128h) };
            let vwsum_64h = unsafe { x86_64::_mm_unpackhi_pd(vwsum, vwsum) };
            unsafe { x86_64::_mm_cvtsd_f64(x86_64::_mm_add_sd(vwsum, vwsum_64h)) }
        });
        handles.push(handle);
    }

    let mut sum: f64 = arr
        .slice(s![arr.len() - leftover..])
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum();
    for handle in handles {
        sum += handle.join().expect("Error Joining a thread");
    }

    sum / arr.len() as f64
}

fn stdev_arr(arr: &ndarray::Array1<f64>) -> f64 {
    var_arr(&arr).sqrt()
}

fn stdev_arr_simd(arr: &ndarray::Array1<f64>) -> f64 {
    var_arr_simd(&arr).sqrt()
}

fn max_arr(arr: &ndarray::Array1<f64>) -> f64 {
    if arr.len() < CORES {
        let mut new_max = f64::MIN;
        for x in arr.iter() {
            if new_max < *x {
                new_max = *x;
            }
        }
        return new_max;
    }
    let indexes = indexing_reps(arr.len());
    let arr = arr.as_ptr() as usize;
    let mut handles = Vec::with_capacity(CORES);
    for (i, rep) in indexes.iter().enumerate() {
        let rep = *rep;
        let local_ptr = arr.clone();
        let handle = thread::spawn(move || {
            let local_ptr = unsafe { (local_ptr as *const f64).add(i) };
            let mut new_max = f64::MIN;
            for r in 0..rep {
                if new_max < unsafe { *local_ptr.add(CORES * r) } {
                    new_max = unsafe { *local_ptr.add(CORES * r) }
                }
            }
            new_max
        });
        handles.push(handle);
    }

    let mut new_max = f64::MIN;
    for handle in handles {
        let new_option = handle.join().expect("Error joining a thread");
        if new_max < new_option {
            new_max = new_option;
        }
    }
    return new_max;
}

fn max_arr_simd(arr: &ndarray::Array1<f64>) -> f64 {
    if arr.len() < CORES * f64::SIMD_LEN {
        return arr.iter().copied().reduce(f64::max).unwrap();
    }

    let (reps, leftover) = chunking_reps(&arr);
    let arr_ptr = arr.as_ptr() as usize;
    let mut handles = Vec::with_capacity(CORES);

    for (i, &rep) in reps.iter().enumerate() {
        let handle = thread::spawn(move || {
            let local_ptr = unsafe { (arr_ptr as *const f64).add(i * f64::SIMD_LEN) };
            let mut vmax = unsafe { x86_64::_mm256_set1_pd(f64::MIN) };

            for r in 0..rep {
                let chunk = unsafe {
                    slice::from_raw_parts(local_ptr.add(CORES * f64::SIMD_LEN * r), f64::SIMD_LEN)
                };
                let chunk = __m256d::from(f64x4::from_slice(chunk));
                vmax = unsafe { x86_64::_mm256_max_pd(vmax, chunk) };
            }

            let mut pvmax = unsafe { x86_64::_mm256_permute2f128_pd(vmax, vmax, 1) };
            vmax = unsafe { x86_64::_mm256_max_pd(vmax, pvmax) };
            pvmax = unsafe { x86_64::_mm256_permute_pd::<5>(vmax) };
            vmax = unsafe { x86_64::_mm256_max_pd(vmax, pvmax) };
            let vmax = unsafe { x86_64::_mm256_extractf128_pd::<0>(vmax) };
            unsafe { x86_64::_mm_cvtsd_f64(vmax) }
        });
        handles.push(handle);
    }

    let mut max: f64 = arr
        .slice(s![arr.len() - leftover..])
        .iter()
        .copied()
        .reduce(f64::max)
        .unwrap_or(f64::MIN);

    for handle in handles {
        let candidate = handle.join().expect("Error Joining a thread");
        if candidate > max {
            max = candidate
        }
    }

    max
}

fn min_arr(arr: &ndarray::Array1<f64>) -> f64 {
    if arr.len() < CORES {
        let mut new_min = f64::MAX;
        for x in arr.iter() {
            if new_min > *x {
                new_min = *x;
            }
        }
        return new_min;
    }
    let indexes = indexing_reps(arr.len());
    let arr = arr.as_ptr() as usize;
    let mut handles = Vec::with_capacity(CORES);
    for (i, rep) in indexes.iter().enumerate() {
        let rep = *rep;
        let local_ptr = arr.clone();
        let handle = thread::spawn(move || {
            let local_ptr = unsafe { (local_ptr as *const f64).add(i) };
            let mut new_min = f64::MAX;
            for r in 0..rep {
                if new_min > unsafe { *local_ptr.add(CORES * r) } {
                    new_min = unsafe { *local_ptr.add(CORES * r) }
                }
            }
            new_min
        });
        handles.push(handle);
    }

    let mut new_min = f64::MAX;
    for handle in handles {
        let new_option = handle.join().expect("Error joining a thread");
        if new_min > new_option {
            new_min = new_option;
        }
    }
    return new_min;
}

fn min_arr_simd(arr: &ndarray::Array1<f64>) -> f64 {
    if arr.len() < CORES * f64::SIMD_LEN {
        return arr.iter().copied().reduce(f64::min).unwrap();
    }

    let (reps, leftover) = chunking_reps(&arr);
    let arr_ptr = arr.as_ptr() as usize;
    let mut handles = Vec::with_capacity(CORES);

    for (i, &rep) in reps.iter().enumerate() {
        let handle = thread::spawn(move || {
            let local_ptr = unsafe { (arr_ptr as *const f64).add(i * f64::SIMD_LEN) };
            let mut vmin = unsafe { x86_64::_mm256_set1_pd(f64::MAX) };

            for r in 0..rep {
                let chunk = unsafe {
                    slice::from_raw_parts(local_ptr.add(CORES * f64::SIMD_LEN * r), f64::SIMD_LEN)
                };
                let chunk = __m256d::from(f64x4::from_slice(chunk));
                vmin = unsafe { x86_64::_mm256_min_pd(vmin, chunk) };
            }

            let mut pvmin = unsafe { x86_64::_mm256_permute2f128_pd(vmin, vmin, 1) };
            vmin = unsafe { x86_64::_mm256_min_pd(vmin, pvmin) };
            pvmin = unsafe { x86_64::_mm256_permute_pd::<5>(vmin) };
            vmin = unsafe { x86_64::_mm256_min_pd(vmin, pvmin) };
            let vmin = unsafe { x86_64::_mm256_extractf128_pd::<0>(vmin) };
            unsafe { x86_64::_mm_cvtsd_f64(vmin) }
        });
        handles.push(handle);
    }

    let mut min: f64 = arr
        .slice(s![arr.len() - leftover..])
        .iter()
        .copied()
        .reduce(f64::min)
        .unwrap_or(f64::MAX);

    for handle in handles {
        let candidate = handle.join().expect("Error Joining a thread");
        if candidate < min {
            min = candidate
        }
    }

    min
}

fn indexing_reps(size: usize) -> Vec<usize> {
    let d = size / CORES;
    let r = size % CORES;
    repeat(d + 1)
        .take(r)
        .chain(repeat(d).take(CORES - r))
        .collect()
}

fn chunking_reps<T: SimdLen>(arr: &ndarray::Array1<T>) -> (Vec<usize>, usize) {
    let chunk_quantity = arr.len() / T::SIMD_LEN;
    let leftover = arr.len() % T::SIMD_LEN;
    let d = chunk_quantity / CORES;
    let r = chunk_quantity % CORES;
    (
        repeat(d + 1)
            .take(r)
            .chain(repeat(d).take(CORES - r))
            .collect(),
        leftover,
    )
}
