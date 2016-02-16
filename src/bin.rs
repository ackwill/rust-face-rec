extern crate image;
extern crate facerec;

use std::path::Path;
use std::fs::File;

use image::GenericImage;

fn main() {
    //network_tests();
    //draw_tests();
    //detect_face();
    avg_hist_test();
}


//TODO: Try and capture only the most dominant fetures of both the image and the trained data that way you can check for a face.
fn avg_hist_test() {
    let trained_vec = facerec::network::avg_hist();

    let img = image::open(&Path::new("background/0140.jpg")).unwrap();
    let sqr = img.to_rgb().sub_image(50, 50, 100, 100).to_image();
    let gimg = facerec::convert_gray_scale(&sqr);

    let hist = facerec::calc_image_hist(&mut facerec::olbp(&gimg), 8, 8);

    let img2 = image::open(&Path::new("kath/2.png")).unwrap();
    let gimg2 = facerec::convert_gray_scale(&img2.to_rgb());

    let hist2 = facerec::calc_image_hist(&mut facerec::olbp(&gimg2), 8, 8);

    println!("{} not face", facerec::compare_hist(&hist, &trained_vec));
    println!("{} face", facerec::compare_hist(&hist2, &trained_vec));

    let mut imagebuff = image::ImageBuffer::new(600, 400);
facerec::drawing::draw_histogram(trained_vec, &mut imagebuff);

    let ref mut fout = File::create(&Path::new("avghisto.png")).unwrap();
    let _ = image::ImageRgb8(imagebuff).save(fout, image::PNG);

}

fn network_tests() {
    //facerec::network::example();
    //facerec::network::test_save();
    //facerec::network::test_load();
    //facerec::network::trian_face_data();
}

fn draw_tests() {
    let mut imagebuff = image::ImageBuffer::new(600, 400);

    let img = image::open(&Path::new("kath/3.png")).unwrap();
    let gimg = facerec::convert_gray_scale(&img.to_rgb());

    let hist = facerec::calc_image_hist(&mut facerec::olbp(&gimg), 8, 8);

    //facerec::drawing::draw_line(799, 799, 0, 0, &mut imagebuff);
    //facerec::drawing::draw_line(0, 799, 799, 0, &mut imagebuff);
    //facerec::drawing::draw_rect(50, 50, 100, 100, &mut imagebuff);
    facerec::drawing::draw_histogram(hist, &mut imagebuff);

    let ref mut fout = File::create(&Path::new("kath3histo.png")).unwrap();
    let _ = image::ImageRgb8(imagebuff).save(fout, image::PNG);
}

fn detect_face() {
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

    face_rec.name_face(0, "Kath".to_string());
    face_rec.name_face(1, "Shane".to_string());

    let mut prediction = face_rec.predict(&gt);
    println!("{:?}", prediction);
    prediction = face_rec.predict(&gt2);
    println!("{:?}", prediction);
}
