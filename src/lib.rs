extern crate image;
extern crate nn;

mod recognizer;

pub mod drawing;
pub mod network;

pub use recognizer::{
    Recognizer,
    convert_gray_scale,
    olbp,
    calc_image_hist,
    compare_hist
};
