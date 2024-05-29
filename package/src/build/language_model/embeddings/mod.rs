use std::fs;
use std::fs::File;
use std::io::{self, Read};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use async_stream::stream;
use futures::stream::Stream;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Embedding {
    pub embedding: Vec<f32>,
}

pub async fn llama_cpp_embedding(text: &str) -> Result<Vec<f32>, io::Error> {

    let docker_embedding_dim = std::env::var("EMBEDDING_CONTEXT_SIZE")
        .as_ref()
        .unwrap()
        .to_string().parse::<usize>().unwrap();

    let input = text.to_string().chars().take(docker_embedding_dim*4).collect::<String>();

    text_embedding_request(&input).await
}


pub async fn text_embedding_request(text: &str) -> Result<Vec<f32>, io::Error> {
    if text.is_empty() {
        return Err(io::Error::new(io::ErrorKind::Other, "Command execution failed: Empty String."));
    }
    let client = reqwest::Client::new();

    let json_data = json!({
        "content": text
    });

    let docker_embedding_endpoint = std::env::var("DOCKER_EMBEDDING_ENDPOINT")
        .as_ref()
        .unwrap()
        .to_string();

    let response = client
        .post(docker_embedding_endpoint)
        .header("Content-Type", "application/json")
        .body(json_data.to_string())
        .send()
        .await;

    if let Ok(response) = response {
        if response.status().is_success() {
            let response = response.json::<Embedding>().await;
            if let Ok(embedding) = response {
                return Ok(embedding.embedding);
            }else{
                return Err(io::Error::new(io::ErrorKind::Other, format!("Command execution failed: {:?}",response)))
            }
        }else{
            return Err(io::Error::new(io::ErrorKind::Other, format!("Command execution failed: {:?}",response)))
        }
    }else{
        return Err(io::Error::new(io::ErrorKind::Other, format!("Command execution failed: {:?}",response)))
    }
}



pub fn extract_embeddings(dataset: Vec<(String, f32)>) -> impl Stream<Item = Result<Value, anyhow::Error>> {
    stream! {
        for (text,label) in dataset {
            match llama_cpp_embedding(&text).await {
                Ok(output) => yield Ok(
                    json!(
                        {
                            "text": Value::from(text),
                            "label": Value::from(label as f64),
                            "embedding": output,
                        }
                    )
                ),
                Err(err) => yield Err(anyhow::anyhow!(format!("An error occurred during embeddings generation: {:?}", err))),
            }
        }
    }
}


pub fn load_llama_cpp_embeddings_from_file(path: &str) -> anyhow::Result<(Vec<Vec<f32>>, Vec<f32>)> {
    // Read the contents of the file
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    // Parse the JSON content
    let json_data: Vec<serde_json::Value> = serde_json::from_str(&contents)?;

    // Initialize vectors to store embeddings and labels
    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();

    // Iterate over each entry in the JSON data
    for entry in json_data {
        // Extract the "embedding" field
        if let Some(embedding) = entry["embedding"].as_array() {
            // Convert the embedding to a vector of f32
            let embedding_vec: Vec<f32> = embedding
                .iter()
                .filter_map(|value| value.as_f64().map(|v| v as f32))
                .collect();

            // Push the embedding to the embeddings vector
            embeddings.push(embedding_vec);
        }

        // Extract the label from the "entry" field (assuming it's an array)
        if let Some(entry_array) = entry["entry"].as_array() {
            // Assuming the label is the second item in the array
            if let Some(label_value) = entry_array.get(1) {
                // Convert the label to f32 and push it to the labels vector
                if let Some(label) = label_value.as_f64() {
                    labels.push(label as f32);
                }
            }
        }
    }

    // Return the result as a tuple
    Ok((embeddings, labels))
}

pub fn load_embeddings_from_file(paths: &[&str]) -> anyhow::Result<(Vec<Vec<f32>>, Vec<f32>)> {

    // Initialize an empty vector to store the predicted values for each path.
    let mut list_sentence_embeddings: Vec<serde_json::Value> = Vec::new();

    // Iterate through each path provided.
    for path in paths {
        println!("Processing file: {}", path);

        // Read the contents of a file that is expected to be present in the directory
        // named "language_model_extract_topics_<path>".
        let sentence_embeddings: serde_json::Value = match fs::read_to_string(format!("language_model_extract_embeddings_{}", path)) {
            // If the file exists and its contents can be parsed as JSON, parse the JSON value
            // and append it to the list_sequence_classification_multi_label_prediction vector.
            Ok(file) => {
                match serde_json::from_str(&file) {
                    Ok(res) => {
                        println!("Successfully read and parsed JSON for file: {}", path);
                        res
                    }
                    // If parsing the JSON value fails, print the error and append a default value
                    // to the list_sequence_classification_multi_label_prediction vector.
                    Err(err) => {
                        println!("Error parsing JSON for file: {}: {:?}", path, err);
                        Default::default()
                    }
                }
            }
            // If the file cannot be read, print the error and append a default value
            // to the list_sequence_classification_multi_label_prediction vector.
            Err(err) => {
                println!("Error parsing JSON for file: {}: {:?}", path, err);
                Default::default()
            }
        };

        // Append the sequence_classification_multi_label_prediction value to the vector.
        list_sentence_embeddings.push(sentence_embeddings);
    }

    let (x_dataset, y_dataset): (Vec<Vec<f32>>, Vec<f32>) = list_sentence_embeddings
        .iter()
        .flat_map(|sentence_embeddings| {
            sentence_embeddings["embeddings"]
                .as_array()
                .unwrap()
                .iter()
                .map(|topics| topics.as_array().unwrap().into_iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<f32>>())
                .zip(
                    sentence_embeddings["dataset"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|entry| {
                            let entry = entry.as_array().unwrap();
                            (
                                entry[0].as_str().unwrap().to_string(),
                                entry[1].as_f64().unwrap() as f32,
                            )
                        })
                        .filter(|(text, _)| text != "empty"),
                ).map(|(topics,(_text,label))|{
                (topics,label)
            })
        })
        .unzip();

    println!("Final x_dataset and y_dataset both contain {} entries.", y_dataset.len());

    Ok((x_dataset,y_dataset))
}
