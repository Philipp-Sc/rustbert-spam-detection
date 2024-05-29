
pub mod build;

use build::classification::*;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use importance::score::Model;


pub async fn fraud_probabilities(texts: &[&str]) ->  anyhow::Result<Vec<f32>> {

    let mut text_embeddings: Vec<Vec<f32>> = Vec::new();
    for text in texts {

       let embedding = build::language_model::embeddings::llama_cpp_embedding(text).await?;
       text_embeddings.push(embedding);

    }

    let model = ClassificationMockModel { label: "./KNNRegressor.bin".to_string(), model_type: ModelType::KNN};
    let x_dataset = text_embeddings.into_par_iter().map(|vec| {
         if vec.len()>0 {
            model.predict(&vec![vec])
         }else{
            vec
         }
       }).collect::<Vec<Vec<f32>>>();

    Ok(x_dataset.into_iter().flatten().collect())
}


