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
        return Err(io::Error::new(io::ErrorKind::Other, "Invalid text embedding request: Empty string!"));
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

    if let Ok(ok_response) = response {
        let debug_response = format!("{:?}",ok_response);
        if ok_response.status().is_success() {
            if let Ok(ref json_response) =  ok_response.json::<Value>().await {
                if let Ok(embedding) = serde_json::from_value::<Embedding>(json_response.clone()) {
                    return Ok(embedding.embedding);
                }else{
                    return Err(io::Error::new(io::ErrorKind::Other, format!("Parsing failed: {:?}\n\n{}\n\n", json_response,text)))
                }
            }else{
                return Err(io::Error::new(io::ErrorKind::Other, format!("Response body is not valid json: {}\n\n{}\n\n", debug_response,text)))
            }
        }else{
            return Err(io::Error::new(io::ErrorKind::Other, format!("Got negative response status: {}\n\n{}\n\n",debug_response,text)))
        }
    }else{
        return Err(io::Error::new(io::ErrorKind::Other, format!("Command execution failed: {:?}\n\n{}\n\n",response,text)))
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

    let mut json_data = Vec::new();
    for line in contents.lines() {
        println!("line: '''{}'''",line);
        let value: serde_json::Value = serde_json::from_str(line)?;
        json_data.push(value);
    }

    // Initialize vectors to store embeddings and labels
    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();

    // Iterate over each entry in the JSON data
    for entry in json_data {
        if let Some(embedding) = entry["embedding"].as_array() {
            // Convert the embedding to a vector of f32
            let embedding_vec: Vec<f32> = embedding
                .iter()
                .filter_map(|value| value.as_f64().map(|v| v as f32))
                .collect();

            // Push the embedding to the embeddings vector
            embeddings.push(embedding_vec);
        }

        if let Some(label) = entry["label"].as_f64() {
            labels.push(label as f32);
        }
    }

    // Return the result as a tuple
    Ok((embeddings, labels))
}
