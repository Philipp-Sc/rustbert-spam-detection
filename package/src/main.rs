use std::env;
use std::fs::File;
use std::io::Write;
use rust_bert_fraud_detection_tools::build::create_embeddings;
use rust_bert_fraud_detection_tools::build::data::{split_vector};
use rust_bert_fraud_detection_tools::build::language_model::embeddings::load_llama_cpp_embeddings_from_file;

const CSV_DATASET: [&str;6] = [
    "./dataset/youtubeSpamCollection.csv",
    "./dataset/enronSpamSubset.csv",
    "./dataset/lingSpam.csv",
    "./dataset/smsspamcollection.csv",
    "./dataset/completeSpamAssassin.csv",
    "./dataset/governance_proposal_spam_likelihood.csv"
];

pub const SENTENCES: [&str;6] = [
    "Lose up to 19% weight. Special promotion on our new weightloss.",
    "Hi Bob, can you send me your machine learning homework?",
    "Don't forget our special promotion: -30% on men shoes, only today!",
    "Hi Bob, don't forget our meeting today at 4pm.",
    "⚠️ FINAL: LAST TERRA PHOENIX AIRDROP 🌎 ✅ CLAIM NOW All participants in this vote will receive a reward..",
    "Social KYC oracle (TYC)  PFC is asking for 20k Luna to build a social KYC protocol.."
    ];
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() <= 1 {
        println!("No command specified.");
        return Ok(());
    }

    let command = &args[1];

    match command.as_str() {
        "train_and_test_text_embedding_knn_regressor" => {train_and_test_text_embedding_knn_regressor(false)?;},
        "train_and_test_text_embedding_knn_regressor_eval" => {train_and_test_text_embedding_knn_regressor(true)?;},

        "save_training_data" => { rust_bert_fraud_detection_tools::build::save_training_data(CSV_DATASET.to_vec()).await?;},

        "generate_embeddings" => {generate_embeddings().await?;},
        "predict" => {

                      let fraud_probabilities = rust_bert_fraud_detection_tools::fraud_probabilities(&SENTENCES).await?;
                      println!("Predictions:\n{:?}",fraud_probabilities);
                      println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
        },
        _ => {panic!()}
    }

    Ok(())
}


fn train_and_test_text_embedding_knn_regressor(eval: bool) -> anyhow::Result<()> {

    let (x_dataset, y_dataset) = load_llama_cpp_embeddings_from_file("./dataset/embedding_dataset.json")?;

    let spam_count = y_dataset.iter().filter(|&&label| label == 1.0).count();
    let ham_count = y_dataset.iter().filter(|&&label| label == 0.0).count();
    let total_count = y_dataset.len();

    println!("Number of Spam entries: {}", spam_count);
    println!("Number of Ham entries: {}", ham_count);
    println!("Total entries: {}", total_count);

    if !eval {
        rust_bert_fraud_detection_tools::build::classification::update_knn_regression_model(&x_dataset,&y_dataset)?;
        rust_bert_fraud_detection_tools::build::classification::test_knn_regression_model(&x_dataset,&y_dataset)?;
    }else {
        let (x_train, x_test) = split_vector(&x_dataset, 0.8);
        let x_train = x_train.to_vec();
        let x_test = x_test.to_vec();
        let (y_train, y_test) = split_vector(&y_dataset, 0.8);
        let y_train = y_train.to_vec();
        let y_test = y_test.to_vec();

        rust_bert_fraud_detection_tools::build::classification::update_knn_regression_model(&x_train,&y_train)?;
        rust_bert_fraud_detection_tools::build::classification::test_knn_regression_model(&x_train,&y_train)?;
        rust_bert_fraud_detection_tools::build::classification::test_knn_regression_model(&x_test,&y_test)?;
    }
    Ok(())
}

async fn generate_embeddings() -> anyhow::Result<()> {

    let mut file = File::options()
        .write(true)
        .append(true)
        .open("embeddings_dataset.json")
        .expect("Failed to open file");

    for dataset_path in CSV_DATASET {
        let embeddings = create_embeddings(vec![&dataset_path]).await?;

        for embedding in embeddings {
            let mut json_data = serde_json::to_string(&embedding).expect("Failed to serialize to JSON");
            json_data.push('\n');
            file.write_all(json_data.as_bytes()).expect("Failed to write to file");
        }
    }
    return Ok(());
}

