use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::iter::repeat;
use std::thread;

const CORES: usize = 6;

#[pymodule]
fn parallel_data_processing_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(stdev, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    Ok(())
}

#[pyfunction]
fn sum(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(sum_arr(&arr))
}

#[pyfunction]
fn mean(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(mean_arr(&arr))
}

#[pyfunction]
fn var(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(var_arr(&arr))
}

#[pyfunction]
fn stdev(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(stdev_arr(&arr))
}

#[pyfunction]
fn max(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(max_arr(&arr))
}

#[pyfunction]
fn min(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.to_owned_array();
    Ok(min_arr(&arr))
}

fn sum_arr(arr: &ndarray::Array1<f64>) -> f64 {
    if arr.len() < CORES {
        return arr.sum();
    }

    let indexes = indexing_reps(arr.len());
    let arr = arr.as_ptr() as usize;
    let mut handles = Vec::with_capacity(CORES);
    for (i, rep) in indexes.iter().enumerate() {
        let rep = *rep;
        let local_ptr = arr.clone();
        let handle = thread::spawn(move || {
            let local_ptr = unsafe{ (local_ptr as *const f64).add(i) };
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

fn mean_arr(arr: &ndarray::Array1<f64>) -> f64 {
    sum_arr(&arr) / arr.len() as f64
}

fn var_arr(arr: &ndarray::Array1<f64>) -> f64 {
    if arr.len() < CORES {
        return arr.var(0.0);
    }

    let len = arr.len() as f64;
    let mean = mean_arr(arr);

    let indexes = indexing_reps(arr.len());
    let arr = arr.as_ptr() as usize;
    let mut handles = Vec::with_capacity(CORES);

    for (i, rep) in indexes.iter().enumerate() {
        let rep = *rep;
        let local_ptr = arr.clone();
        let handle = thread::spawn(move || {
            let local_ptr = unsafe{ (local_ptr as *const f64).add(i) };
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
    sum / len
}

fn stdev_arr(arr: &ndarray::Array1<f64>) -> f64 {
    var_arr(&arr).sqrt()
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
            let local_ptr = unsafe{ (local_ptr as *const f64).add(i) };
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
            let local_ptr = unsafe{ (local_ptr as *const f64).add(i) };
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

fn indexing_reps(size: usize) -> Vec<usize> {
    let d = size / CORES;
    let r = size % CORES;
    repeat(d + 1)
        .take(r)
        .chain(repeat(d).take(CORES - r))
        .collect()
}
