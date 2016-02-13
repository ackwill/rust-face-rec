extern crate image;
extern crate facerec;

use std::path::Path;

fn main() {
    let mut face_rec = facerec::Recognizer::new();

    let mut images = Vec::new();
    let mut labels = Vec::new();

    //Kath 0
    let kath1 = image::open(&Path::new("kath/1.png")).unwrap();
    let gk1 = facerec::convert_gray_scale(&kath1.to_rgb());

    let kath2 = image::open(&Path::new("kath/2.png")).unwrap();
    let gk2 = facerec::convert_gray_scale(&kath2.to_rgb());

    let kath3 = image::open(&Path::new("kath/3.png")).unwrap();
    let gk3 = facerec::convert_gray_scale(&kath3.to_rgb());

    //Shane 1
    let shane1 = image::open(&Path::new("shane/2.png")).unwrap();
    let gsh1 = facerec::convert_gray_scale(&shane1.to_rgb());

    let shane2 = image::open(&Path::new("shane/3.png")).unwrap();
    let gsh2 = facerec::convert_gray_scale(&shane2.to_rgb());

    let shane3 = image::open(&Path::new("shane/4.png")).unwrap();
    let gsh3 = facerec::convert_gray_scale(&shane3.to_rgb());

    let testi = image::open(&Path::new("shane/6.png")).unwrap();
    let gt = facerec::convert_gray_scale(&testi.to_rgb());

    let test2 = image::open(&Path::new("kath/4.png")).unwrap();
    let gt2 = facerec::convert_gray_scale(&test2.to_rgb());

    images.push(gk1.clone());
    images.push(gk2.clone());
    images.push(gk3.clone());
    images.push(gsh1.clone());
    images.push(gsh2.clone());
    images.push(gsh3.clone());

    labels.push(0);
    labels.push(0);
    labels.push(0);
    labels.push(1);
    labels.push(1);
    labels.push(1);

    face_rec.train(images, labels);

    face_rec.name_face(1, "Shane".to_string());
    face_rec.name_face(0, "Kath".to_string());

    let mut prediction = face_rec.predict(&gt);
    println!("{:?}", prediction);
    prediction = face_rec.predict(&gt2);
    println!("{:?}", prediction);
}
