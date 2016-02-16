use image;
use image::{ImageBuffer, GrayImage, RgbImage, GenericImage};
use std::f64;

pub mod statslib;

pub type Histogram = Vec<f64>;

pub struct TrainedFace {
    pub histograms: Vec<Histogram>,
    label: usize,
    name: String,
    mean: f64,
}

impl TrainedFace {
    fn new(id: usize) -> TrainedFace {
        TrainedFace {
            histograms: Vec::new(),
            name: String::new() ,
            label: id,
            mean: 0.0
        }
    }
}

pub struct Recognizer {
    //histograms: Vec<ImageHistogram>,
    trained_data: Vec<TrainedFace>,
}

impl Recognizer {
    pub fn new() -> Recognizer {
        Recognizer {
            //histograms: Vec::new(),
            trained_data: Vec::new(),
        }
    }

    pub fn predict(&self, img: &GrayImage) -> (usize, f64, String) {
        let confidence: f64;
        let mut best_fit_value: f64 = f64::MAX;
        let mut label: usize = 0;

        let mut lbp = olbp(&img);
        let src_hist = calc_image_hist(&mut lbp, 8, 8);

        // Find closest face match
        for face in &self.trained_data {
            for hist in &face.histograms {
                let value = compare_hist(&src_hist, &hist);
                if value < best_fit_value {
                    best_fit_value = value;
                    label = face.label.clone();
                }
            }
        }

        // TODO: Check if value is in acceptable range


        // Get prediction confidence and name
        confidence = if best_fit_value != 0.0 {best_fit_value / self.trained_data[label].mean * 100.0} else {100.00};
        let name = self.trained_data[label].name.clone();

    (label, confidence, name)
    }

    pub fn train(&mut self, images: Vec<GrayImage>, img_labels: Vec<usize>) {
        if images.len() != img_labels.len() {
            panic!("Images and labels vector are not the same size");
        }

        // Calculate a histogram for each image
        for i in 0..images.len() {
            let mut img = olbp(&images[i]);
            // TODO: Make cell_x, cell_y global or resettable
            let img_hist = calc_image_hist(&mut img, 8, 8);
            let id = img_labels[i].clone();
            if id == self.trained_data.len() {
                self.trained_data.push(TrainedFace::new(id));
            }
            self.trained_data[id].histograms.push(img_hist);
        }

        // Calculate mean prediction value for each face
        for face in &mut self.trained_data {
            let n = face.histograms.len();
            let mut cmpr_values = Vec::new();
            for i in 0..n {
                for j in (i+1)..n {
                    let value = compare_hist(&face.histograms[i], &face.histograms[j]);
                    cmpr_values.push(value);
                }
            }

            face.mean = statslib::mean(&cmpr_values);
        }
    }

    pub fn name_face(&mut self, id: usize, name: String) {
        self.trained_data[id].name = name;
    }

    pub fn get_name(&mut self, id: usize) -> String {
        self.trained_data[id].name.clone()
    }
/*
    pub fn load(&self, file: &Path) {
        unimplemented!();
        let img = image::open(&Path::new("tom.jpg")).unwrap();
        let gray_img = convert_gray_scale(&img.to_rgb());
        let lbp = olbp(&gray_img);

        let ref mut fout = File::create(&Path::new("tom.png")).unwrap();

        let _ = image::ImageLuma8(lbp).save(fout, image::PNG).unwrap();
    }

    pub fn save(&self, out: &Path) {
        unimplemented!();
    }
*/

}

pub fn olbp(img: &GrayImage) -> GrayImage {

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

    //let norm_hist: Histogram = statslib::normalize(hist);

    //norm_hist
    hist
}

pub fn calc_image_hist(img: &mut GrayImage, cell_x: u32, cell_y: u32) -> Histogram {
    let (cols, rows) = img.dimensions();

    let cell_width = cols / cell_x;
    let cell_height = rows / cell_y;

    let mut image_hist: Histogram = Vec::new();

    for x in 0..cell_x {
        for y in 0..cell_y {
            // TODO: Fiture out what 'a does !!!
            let cell = img.sub_image::<'a>(x*cell_width, y*cell_height, cell_width, cell_height).to_image();
            let mut cell_hist = calc_cell_hist(&cell);
            image_hist.append(&mut cell_hist);
        }
    }
    image_hist = statslib::normalize(image_hist);
    image_hist
}

// use Chisquare alternative
pub fn compare_hist(hist_1: &Histogram, hist_2: &Histogram) -> f64 {
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

    let r_scale = 0.299;
    let g_scale = 0.587;
    let b_scale = 0.114;

    for x in 0..cols {
        for y in 0..rows {
            let c_pixel = img.get_pixel(x, y);
            let gray_pixel = image::Luma([(r_scale*c_pixel[0] as f64 + g_scale*c_pixel[1] as f64 + b_scale*c_pixel[2] as f64) as u8]);
            gray_img.put_pixel(x, y, gray_pixel);
        }
    }
    gray_img
}
