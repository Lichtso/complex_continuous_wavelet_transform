use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct CCWT {
    input_width: usize,
    input_sample_count: usize,
    input_padding: usize,
    output_width: usize,
    output_sample_count: usize,
    output_padding: usize,
    half_width: f64,
    scale_factor: f64,
    padding_correction: f64,
    input_freq_domain: Vec<Complex<f64>>,
    output_time_domain: Vec<Complex<f64>>,
    output_freq_domain: Vec<Complex<f64>>,
    output_plan: std::sync::Arc<rustfft::FFT<f64>>
}

#[wasm_bindgen]
pub fn new_ccwt(input: &mut [f32], input_padding: usize, output_width: usize) -> CCWT {
    assert!(output_width <= input.len());
    let input_sample_count = input.len()+2*input_padding;
    let output_padding = input_padding*output_width/input.len();
    let output_sample_count = output_width+2*output_padding; // output_width*input_sample_count/input.len();
    let mut output_planner = rustfft::FFTplanner::<f64>::new(true);
    let mut ccwt = CCWT {
        input_width: input.len(),
        input_sample_count: input_sample_count,
        input_padding: input_padding,
        output_width: output_width,
        output_sample_count: output_sample_count,
        output_padding: output_padding,
        half_width: (input_sample_count as f64)*0.5,
        scale_factor: 1.0/(input_sample_count as f64),
        padding_correction: (input_sample_count as f64)/(input.len() as f64),
        input_freq_domain: vec![Complex::zero(); input_sample_count],
        output_time_domain: vec![Complex::zero(); output_sample_count],
        output_freq_domain: vec![Complex::zero(); output_sample_count],
        output_plan: Arc::clone(&output_planner.plan_fft(output_sample_count))
    };
    let mut input_time_domain = vec![Complex::zero(); input_sample_count];
    for x in 0..ccwt.input_width {
        input_time_domain[ccwt.input_padding+x].re = input[x] as f64;
    }
    let mut input_planner = rustfft::FFTplanner::<f64>::new(false);
    let input_plan = input_planner.plan_fft(ccwt.input_sample_count);
    input_plan.process(&mut input_time_domain, &mut ccwt.input_freq_domain);
    ccwt
}

macro_rules! gabor_wavelet {
    ( $ccwt:ident, $frequency:ident, $deviation:ident, $output_x:ident, $input_x:expr, $operator:tt ) => {
        let f: f64 = $ccwt.half_width-((($input_x as f64)-$frequency).abs()-$ccwt.half_width).abs();
        $ccwt.output_freq_domain[$output_x] $operator Complex::new(-f*f*$deviation, 0.0).exp()*$ccwt.input_freq_domain[$input_x]*$ccwt.scale_factor;
    };
}

#[wasm_bindgen]
pub fn get_transformed_frequency(ccwt: &mut CCWT, output: &mut [f32], mut frequency: f64, derivative: f64) {
    frequency *= ccwt.padding_correction;
    let heisenberg_gabor_limit: f64 = (1.0/(4.0*std::f64::consts::PI)).sqrt(); // 0.28209479177
    let deviation = 1.0/(derivative*ccwt.padding_correction*(ccwt.output_sample_count as f64)*heisenberg_gabor_limit);
    for output_x in 0..ccwt.output_sample_count {
        gabor_wavelet!(ccwt, frequency, deviation, output_x, output_x, =);
    }
    if ccwt.output_sample_count < ccwt.input_sample_count {
        let rest = ccwt.input_sample_count%ccwt.output_sample_count;
        let cut_index = ccwt.input_sample_count-rest;
        for chunk_index in (ccwt.output_sample_count..cut_index).step_by(ccwt.output_sample_count) {
            for output_x in 0..ccwt.output_sample_count {
                gabor_wavelet!(ccwt, frequency, deviation, output_x, chunk_index+output_x, +=);
            }
        }
        for output_x in 0..rest {
            gabor_wavelet!(ccwt, frequency, deviation, output_x, cut_index+output_x, +=);
        }
    }
    ccwt.output_plan.process(&mut ccwt.output_freq_domain, &mut ccwt.output_time_domain);
    for x in 0..ccwt.output_width {
        let sample = ccwt.output_time_domain[ccwt.output_padding+x];
        output[x*2  ] = sample.re as f32;
        output[x*2+1] = sample.im as f32;
    }
}
