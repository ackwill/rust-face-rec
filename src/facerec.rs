use super::image;
use super::image::{ImageBuffer, GrayImage, RgbImage, GenericImage};
use std::fs::File;
use std::path::Path;

pub type Histogram = Vec<f64>;

struct ImageHistogram {
    histogram: Histogram,
    label: String,
}

pub struct FaceRec {
    histograms: Vec<ImageHistogram>,
}

impl FaceRec {
    pub fn new() -> FaceRec {
        FaceRec {
            histograms: Vec::new(),
        }
    }

    pub fn predict(&self, img: &GrayImage) -> String {
        //TODO: Make this value correct
        // Do that by calculating a minimum value for each training data. That would be equal to the average prediction value/
        // for when all like images are compared together. 
        let mut best_fit: f64 = 10000.0;
        let mut label: String = "".to_string();

        let mut lbp = FaceRec::olbp(&img);
        let src_hist = FaceRec::calc_image_hist(&mut lbp, 8, 8);

        for i in &self.histograms {
            let value = FaceRec::compare_hist(&src_hist, &i.histogram);
            if value < best_fit {
                best_fit = value;
                label = i.label.clone();
            }
        }
        label

    }

    pub fn train(&mut self, images: Vec<GrayImage>, img_labels: Vec<String>) {
        if images.len() != img_labels.len() {
            panic!("Images and labels vector are not the same size");
        }

        for i in 0..images.len() {
            let mut img = FaceRec::olbp(&images[i]);
            // TODO: Make cell_x, cell_y global or resettable
            let img_hist = FaceRec::calc_image_hist(&mut img, 8, 8);
            let name = img_labels[i].clone();
            let histo = ImageHistogram{histogram: img_hist, label: name};
            self.histograms.push(histo);
        }
    }

    pub fn load(&self, file: &Path) {
        let img = image::open(&Path::new("tom.jpg")).unwrap();
        let gray_img = FaceRec::convert_gray_scale(&img.to_rgb());
        let lbp = FaceRec::olbp(&gray_img);

        let ref mut fout = File::create(&Path::new("tom.png")).unwrap();

        let _ = image::ImageLuma8(lbp).save(fout, image::PNG).unwrap();
    }

    pub fn save(&self, out: &Path) {

    }

    fn olbp(img: &GrayImage) -> GrayImage {

        let (cols, rows) = img.dimensions();

        let mut imgbuff = ImageBuffer::new(cols-2, rows-2);

        for x in 1..cols-1 {
            for y in 1..rows-1 {
                let center = img.get_pixel(x, y);
                let mut code: u8 = 0;
                code |= if img.get_pixel(x-1, y-1)[0] >= center[0] {1 << 7} else {0 << 7};
                code |= if img.get_pixel(x-1, y)[0]   >= center[0] {1 << 6} else {0 << 6};
                code |= if img.get_pixel(x-1, y+1)[0] >= center[0] {1 << 5} else {0 << 5};
                code |= if img.get_pixel(x, y+1)[0]   >= center[0] {1 << 4} else {0 << 4};
                code |= if img.get_pixel(x+1, y+1)[0] >= center[0] {1 << 3} else {0 << 3};
                code |= if img.get_pixel(x+1, y)[0]   >= center[0] {1 << 2} else {0 << 2};
                code |= if img.get_pixel(x+1, y-1)[0] >= center[0] {1 << 1} else {0 << 1};
                code |= if img.get_pixel(x, y-1)[0]   >= center[0] {1 << 0} else {0 << 0};
                imgbuff.put_pixel(x-1, y-1, image::Luma([code]));
            }
        }
        imgbuff
    }

    fn calc_cell_hist(img: &GrayImage) -> Histogram {
        let (cols, rows) = img.dimensions();
        let img_vec = img.to_vec();
        let mut hist: Histogram = vec![0.0; 256];

        for x in 0..cols as usize {
            for y in 0..rows as usize {
                let index = img_vec[y*cols as usize + x] as usize;
                hist[index] += 1.0;
            }
        }

        let norm_hist: Histogram = FaceRec::normalize(hist);

        norm_hist
    }

    fn calc_image_hist(img: &mut GrayImage, cell_x: u32, cell_y: u32) -> Histogram {
        let (cols, rows) = img.dimensions();

        let cell_width = cols / cell_x;
        let cell_height = rows / cell_y;

        let mut image_hist: Histogram = Vec::new();

        for x in 0..cell_x {
            for y in 0..cell_y {
                // Fiture out what 'a does !!!
                let cell = img.sub_image::<'a>(x*cell_width, y*cell_height, cell_width, cell_height).to_image();
                let mut cell_hist = FaceRec::calc_cell_hist(&cell);
                image_hist.append(&mut cell_hist);
            }
        }
        image_hist
    }

    fn compare_hist(hist_1: &Histogram, hist_2: &Histogram) -> f64 {
        if hist_1.len() != hist_2.len() {
            panic!("Histograms are not comparable.");
        }

        let mut result = 0.0;

        for i in 0..hist_1.len() {
            let diff = hist_1[i] - hist_2[i];
            let sum = hist_1[i] + hist_2[i];
            let value = if sum > 0.0 {2.0 * (diff*diff) / sum} else {0.0};
            result += value;
        }
        result
    }

    pub fn convert_gray_scale(img: &RgbImage) -> GrayImage {
        let (cols, rows) = img.dimensions();

        let mut gray_img = GrayImage::new(cols, rows);

        for x in 0..cols {
            for y in 0..rows {
                let c_pixel = img.get_pixel(x, y);
                let gray_pixel = image::Luma([(0.299*c_pixel[0] as f64 + 0.587*c_pixel[1] as f64 + 0.114*c_pixel[2] as f64) as u8]);
                gray_img.put_pixel(x, y, gray_pixel);
            }
        }
        gray_img
    }

    fn normalize(hist: Histogram) -> Histogram {
        let mut norm_hist: Histogram = Vec::new();
        let mut max: f64 = 0.0;
        for x in &hist {
            if *x > max {
                max = x.clone();
            }
        }
        for x in &hist {
            norm_hist.push(*x / max);
        }
        norm_hist
    }
}
