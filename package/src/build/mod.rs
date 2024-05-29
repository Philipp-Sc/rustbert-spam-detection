use serde_json;
use std::fs::File;
use std::io::Write;
use futures::Stream;
use serde_json::{Value};

pub mod data;
pub mod classification;
pub mod language_model;

use data::read_datasets;


pub async fn save_training_data(dataset_paths: Vec<&str>) -> anyhow::Result<()> {

    let dataset: Vec<(String,f32)> = read_datasets(&dataset_paths)?;


    let spam_count = dataset.iter().filter(|x| x.1 >= 0.5).count();
    let ham_count = dataset.iter().filter(|x| x.1 < 0.5).count();
    let total_count = dataset.len();

    let spam_percentage = (spam_count as f64 / total_count as f64) * 100.0;
    let ham_percentage = (ham_count as f64 / total_count as f64) * 100.0;

    println!("Spam count: {}", spam_count);
    println!("Ham count: {}", ham_count);
    println!("===================");
    println!("Total count: {}", total_count);
    println!("===================");
    println!("Spam percentage: {:.2}%", spam_percentage);
    println!("Ham percentage: {:.2}%\n", ham_percentage);


    let dataset_view: Vec<(&str,&f32)> = dataset.iter().map(|(text,label)| (text.as_str(),label)).collect();

    let json_data = serde_json::to_string_pretty(&dataset_view).expect("Failed to serialize to JSON");

    let mut file = File::create("raw_dataset.json").expect("Failed to create file");
    file.write_all(json_data.as_bytes()).expect("Failed to write to file");

    Ok(())
}


pub async fn create_embeddings(dataset_paths: Vec<&str>) -> anyhow::Result<(usize,impl Stream<Item = Result<Value, anyhow::Error>>)> {

    let dataset: Vec<(String,f32)> = read_datasets(&dataset_paths)?;

    let spam_count = dataset.iter().filter(|x| x.1 >= 0.5).count();
    let ham_count = dataset.iter().filter(|x| x.1 < 0.5).count();
    let total_count = dataset.len();

    let spam_percentage = (spam_count as f64 / total_count as f64) * 100.0;
    let ham_percentage = (ham_count as f64 / total_count as f64) * 100.0;

    println!("Spam count: {}", spam_count);
    println!("Ham count: {}", ham_count);
    println!("===================");
    println!("Total count: {}", total_count);
    println!("===================");
    println!("Spam percentage: {:.2}%", spam_percentage);
    println!("Ham percentage: {:.2}%\n", ham_percentage);

    let embeddings_iter = language_model::embeddings::extract_embeddings(dataset);

    Ok((total_count,embeddings_iter))
}
