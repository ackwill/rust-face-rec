use image;
use std::cmp;

use recognizer;

pub fn draw_line(x1: usize, y1: usize, x2: usize, y2: usize, src: &mut image::RgbImage) {
    let start;
    let stop;
    if x1 == x2 {
        start = cmp::min(y1, y2);
        stop = cmp::max(y1, y2);
        for y in start..stop+1 {
            let pixel = image::Rgb([0,0,255]);
            src.put_pixel(x1 as u32, y as u32, pixel);
        }
    } else {
        start = cmp::min(x1, x2);
        stop = cmp::max(x1, x2);
        for x in start..stop+1 {
            let slope: isize = (y2 as isize - y1 as isize) / (x2 as isize - x1 as isize);
            let y: isize = slope*(x-x1) as isize + y1 as isize;
            let pixel = image::Rgb([0,0,255]);
            src.put_pixel(x as u32, y as u32, pixel);
        }
    }
}

pub fn draw_rect(x: usize, y: usize, width: usize, height: usize, src: &mut image::RgbImage) {
    draw_line(x, y, x+width-1, y, src);
    draw_line(x+width-1, y, x+width-1, y+height-1, src);
    draw_line(x+width-1, y+height-1, x, y+height-1, src);
    draw_line(x, y+height-1, x, y, src);
}

pub fn draw_histogram(hist: Vec<f64>, src: &mut image::RgbImage) {
    let width = src.width();
    let height = src.height() as usize - 1 ;

    let bin: f64 = width as f64/ hist.len() as f64;

    let norm_data = recognizer::statslib::norm_range(&hist, height as f64, 0.0);

    for x in 1..norm_data.len() {
        let x1 = ((x-1) as f64 * bin) as usize;
        let y1 = height - norm_data[x-1] as usize;
        let x2 = (x as f64 * bin) as usize;
        let y2 = height - norm_data[x] as usize;
        //println!("{} {} {} {} hist: {} {}", x1, y1, x2, y2, norm_data[x-1], norm_data[x]);
        draw_line(x1, y1, x2, y2, src);
    }
}
