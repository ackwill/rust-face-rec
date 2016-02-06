extern crate image;

mod facerec;

use std::path::Path;

fn main() {
    let mut face_rec = facerec::FaceRec::new();

    let mut images = Vec::new();
    let mut labels = Vec::new();

    let img = image::open(&Path::new("tom.jpg")).unwrap();
    let gray_img = facerec::FaceRec::convert_gray_scale(&img.to_rgb());
    let img2 = image::open(&Path::new("4.png")).unwrap();
    let gr = facerec::FaceRec::convert_gray_scale(&img2.to_rgb());

    let test_img = image::open(&Path::new("3.png")).unwrap();
    let test_gr = facerec::FaceRec::convert_gray_scale(&test_img.to_rgb());

    images.push(gray_img.clone());
    images.push(gr.clone());

    labels.push("Tom".to_string());
    labels.push("Shane".to_string());


    face_rec.train(images, labels);
    let mut prediction = face_rec.predict(&gray_img);
    println!("{:?}", prediction);
    prediction = face_rec.predict(&test_gr);
    println!("{:?}", prediction);
}
