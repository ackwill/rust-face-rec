use image;
use nn::{NN, HaltCondition};

use recognizer;

use std::error::Error;
use std::path::Path;
use std::fs::File;
use std::io::prelude::*;


pub fn avg_hist() -> Vec<f64> {
    let mut avg_hist: Vec<f64> = Vec::new();
    for i in 0..(64*256) {
        avg_hist.push(0.0);
    }

    let mut count = 0;
    for i in 0..400 {
        let name = format!("faces/output{}.png", i);

        let img = image::open(&Path::new(&name)).unwrap();
        let gimg = recognizer::convert_gray_scale(&img.to_rgb());

        let hist = recognizer::calc_image_hist(&mut recognizer::olbp(&gimg), 8, 8);

        for j in 0..hist.len() {
            avg_hist[j] += hist[j];
        }
        count += 1;
    }

    for i in 0..avg_hist.len() {
        avg_hist[i] = avg_hist[i] / count as f64;
    }
    
    avg_hist
}

/*
pub fn trian_face_data() {
    let mut data: [(Vec<f64>, Vec<f64>); 851] = [
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]), (vec![], vec![]),
    (vec![], vec![])
    ];



    let mut iter = 0;
    for i in 0..400 {
        let name = format!("faces/output{}.png", i);

        println!("{}", name);

        let img = image::open(&Path::new(&name)).unwrap();
        let gimg = recognizer::convert_gray_scale(&img.to_rgb());

        let hist = recognizer::calc_image_hist(&mut recognizer::olbp(&gimg), 8, 8);

        data[iter] = (hist, vec![1f64]);
        iter += 1;
    }
    for i in 100..551 {
        let name = format!("background/0{}.jpg", i);

        println!("{}", name);

        let img = image::open(&Path::new(&name)).unwrap();
        let gimg = recognizer::convert_gray_scale(&img.to_rgb());

        let hist = recognizer::calc_image_hist(&mut recognizer::olbp(&gimg), 8, 8);

        data[iter] = (hist, vec![0f64]);

        iter += 1;
    }

    let mut net = NN::new(&[16384, 3, 5, 1]);

    net.train(&data).log_interval(Some(100)).go();

    save_network(net, &Path::new("network.json"));


}

*/

fn load_network(network_path: &str) -> NN{
    let path = Path::new(network_path);
    let display = path.display();

    let mut net: NN;

    let mut file = match File::open(&path) {
        Err(why) => panic!("couldn't open {}: {}", display, Error::description(&why)),
        Ok(file) => file,
    };


    let mut s = String::new();
    match file.read_to_string(&mut s) {
        Err(why) => panic!("couldn't read {}: {}", display,
                                                   Error::description(&why)),
        Ok(_) => net = NN::from_json(&s),
    }

    net
}

fn save_network(net: NN, path: &Path) {
    //let path = Path::new("network.json");
    let display = path.display();

    // Open a file in write-only mode, returns `io::Result<File>`
    let mut file = match File::create(path) {
        Err(why) => panic!("couldn't create {}: {}",
                           display,
                           Error::description(&why)),
        Ok(file) => file,
    };

    match file.write_all(net.to_json().as_bytes()) {
        Err(why) => {
            panic!("couldn't write to {}: {}", display,
                                               Error::description(&why))
        },
        Ok(_) => println!("successfully wrote to {}", display),
    }
}

pub fn test_load() {
    let net = load_network("network.json");

    let kath3 = image::open(&Path::new("kath/3.png")).unwrap();
    let gk3 = recognizer::convert_gray_scale(&kath3.to_rgb());

    let hist1 = recognizer::calc_image_hist(&mut recognizer::olbp(&gk3), 8, 8);

    let result = net.run(&hist1);
    println!("{:?}", result);

}

//TODO: Impliment draw_histogram so that we can compare testdata to input data

pub fn test_save() {
    let kath1 = image::open(&Path::new("kath/1.png")).unwrap();
    let gk1 = recognizer::convert_gray_scale(&kath1.to_rgb());

    let kath2 = image::open(&Path::new("kath/2.png")).unwrap();
    let gk2 = recognizer::convert_gray_scale(&kath2.to_rgb());

    let kath3 = image::open(&Path::new("kath/3.png")).unwrap();
    let gk3 = recognizer::convert_gray_scale(&kath3.to_rgb());

    let hist1 = recognizer::calc_image_hist(&mut recognizer::olbp(&gk1), 8, 8);
    let hist2 = recognizer::calc_image_hist(&mut recognizer::olbp(&gk2), 8, 8);
    let hist3 = recognizer::calc_image_hist(&mut recognizer::olbp(&gk3), 8, 8);

    let shane1 = image::open(&Path::new("shane/2.png")).unwrap();
    let gsh1 = recognizer::convert_gray_scale(&shane1.to_rgb());

    let shane2 = image::open(&Path::new("shane/3.png")).unwrap();
    let gsh2 = recognizer::convert_gray_scale(&shane2.to_rgb());

    let shane3 = image::open(&Path::new("shane/4.png")).unwrap();
    let gsh3 = recognizer::convert_gray_scale(&shane3.to_rgb());

    let hist4 = recognizer::calc_image_hist(&mut recognizer::olbp(&gsh1), 8, 8);
    let hist5 = recognizer::calc_image_hist(&mut recognizer::olbp(&gsh2), 8, 8);
    let hist6 = recognizer::calc_image_hist(&mut recognizer::olbp(&gsh3), 8, 8);

    let test = hist4.clone();

    let examples = [
    (hist1, vec![1f64]),
    (hist2, vec![1f64]),
    (hist3, vec![1f64]),
    (hist4, vec![0f64]),
    (hist5, vec![0f64]),
    (hist6, vec![0f64])
    ];

    let mut net = NN::new(&[16384, 3, 1]);

    net.train(&examples).log_interval(Some(100)).go();

    save_network(net, &Path::new("network.json"));
}

pub fn example() {

    // create examples of the XOR function
    // the network is trained on tuples of vectors where the first vector
    // is the inputs and the second vector is the expected outputs
    let examples = [
    (vec![0f64, 0f64], vec![0f64]),
    (vec![0f64, 1f64], vec![1f64]),
    (vec![1f64, 0f64], vec![1f64]),
    (vec![1f64, 1f64], vec![0f64]),
    ];

    // create a new neural network by passing a pointer to an array
    // that specifies the number of layers and the number of nodes in each layer
    // in this case we have an input layer with 2 nodes, one hidden layer
    // with 3 nodes and the output layer has 1 node
    let mut net = NN::new(&[2, 3, 1]);

    // train the network on the examples of the XOR function
    // all methods seen here are optional except go() which must be called to begin training
    // see the documentation for the Trainer struct for more info on what each method does
    net.train(&examples)
    .halt_condition( HaltCondition::Epochs(10000) )
    .log_interval( Some(100) )
    .momentum( 0.1 )
    .rate( 0.3 )
    .go();

    // evaluate the network to see if it learned the XOR function
    for &(ref inputs, ref outputs) in examples.iter() {
        let results = net.run(inputs);
        let (result, key) = (results[0].round(), outputs[0]);
        assert!(result == key);
    }
}
