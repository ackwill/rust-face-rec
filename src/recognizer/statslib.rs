//use std::f64::consts;

pub fn normalize(data: Vec<f64>) -> Vec<f64> {
    let mut norm_data: Vec<f64> = Vec::new();
    let mut max: f64 = 0.0;
    for x in &data {
        if *x > max {
            max = x.clone();
        }
    }
    for x in &data {
        norm_data.push(*x / max);
    }
    norm_data
}

pub fn norm_range(data: &Vec<f64>, high: f64, low: f64) -> Vec<f64> {
    let mut norm_data: Vec<f64> = Vec::new();
    let mut max: f64 = 0.0;
    let mut min: f64 = 0.0;

    for x in data {
        if *x > max {
            max = x.clone();
        }
        if *x < min {
            min = x.clone();
        }
    }

    let m = (low-high)/(min-max);
    let b = ((min*high) - (max*low))/(min-max);

    for x in data {
        norm_data.push(m* *x + b);
    }

    norm_data
}

pub fn mean(values: &Vec<f64>) -> f64 {
    let mut mean = 0.0;
    for x in values {
        mean += *x as f64;
    }
    mean / values.len() as f64
}
/*
pub fn norm_pdf(x: f64, std: f64, mean: f64) -> f64{
    (-1.0*(x-mean).powi(2)/(2.0*std.powi(2))).exp() / (std*(2.0*consts::PI).sqrt())
}

//TODO: Make these fucnions have integer traits
pub fn stdeviation(values: &Vec<f64>, mean: &f64) -> f64 {
    let mut stdeviation = 0.0;
    for i in values {
        stdeviation += (i - *mean).powi(2);
    }
    stdeviation /= (values.len() as f64) - 1.0;
    stdeviation
}
*/
