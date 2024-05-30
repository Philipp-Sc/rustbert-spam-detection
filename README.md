<div align="center">
<img src="https://img.shields.io/github/languages/top/Philipp-Sc/llm-fraud-detection">
<img src="https://img.shields.io/github/repo-size/Philipp-Sc/llm-fraud-detection">
<img src="https://img.shields.io/github/commit-activity/m/Philipp-Sc/llm-fraud-detection">
<img src="https://img.shields.io/github/license/Philipp-Sc/llm-fraud-detection">
</div>

# llm-fraud-detection
Robust semi-supervised fraud detection using Rust native NLP pipelines.
# About
**llm-fraud-detection** relies on [llama.cpp](https://github.com/ggerganov/llama.cpp) to generate **text embeddings** from a given text to predict the fraud likelihood.

The training data is generated from a diverse collection of commonly used spam/ham datasets:
- Ling Spam,
- Enron Spam,
- Spam Assassin Dataset,
- SMS Spam Collection,
- Youtube Spam,
- Crypto Governance Proposals.

**llm-fraud-detection** archives state-of-the-art performance without fine tuning the LLMs directly, instead the outputs of the LLMs (embeddings) are trained on and used for the spam/ham classification task.

# Use

Git clone and train the required models. Edit the docker-compose.yml first.

```bash
docker-compose up -d --build
```
(the training data is provided in this repository, the models are not due to size limitations)

Add to your `Cargo.toml` manifest:

```ini
[dependencies]
rust_fraud_detection_tools = { git="https://github.com/Philipp-Sc/llm-fraud-detection.git" }
```

Predict fraud/ham:
```rust

pub const SENTENCES: [&str;6] = [
    "Lose up to 19% weight. Special promotion on our new weightloss.",
    "Hi Bob, can you send me your machine learning homework?",
    "Don't forget our special promotion: -30% on men shoes, only today!",
    "Hi Bob, don't forget our meeting today at 4pm.",
    "⚠️ FINAL: LAST TERRA PHOENIX AIRDROP 🌎 ✅ CLAIM NOW All participants in this vote will receive a reward..",
    "Social KYC oracle (TYC)  PFC is asking for 20k Luna to build a social KYC protocol.."
    ];


fn main() -> anyhow::Result<()> {

    let fraud_probabilities: Vec<f32> = rust_fraud_detection_tools::fraud_probabilities(&SENTENCES)?;
    println!("Predictions:\n{:?}",fraud_probabilities);
    println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
    Ok(())
}

```
```
Predictions:
[1.0, 0.0, 0.6536912, 0.0, 0.99759775, 0.17467633]
Labels:
[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
```

# Architecture

## Features

### KNN Regressor on Text Embeddings (llama.cpp) ⭐
Single fraud likelihood prediction based on text embeddings.
Model: [uae-large-v1_fp32.gguf](https://huggingface.co/ChristianAzinn/uae-large-v1-gguf)

<details>
<summary> <b>Expand to display the full evaluation (F-Score = 0.94) </b> </summary>

```
Performance on the training data (80%)
```
```rust
Threshold >= 0.1: True Positive = 5978, False Positive = 732, Precision = 0.891, Recall = 1.000, F-Score = 0.942
Threshold >= 0.2: True Positive = 5978, False Positive = 110, Precision = 0.982, Recall = 1.000, F-Score = 0.991
Threshold >= 0.3: True Positive = 5977, False Positive = 72, Precision = 0.988, Recall = 0.999, F-Score = 0.994
Threshold >= 0.4: True Positive = 5975, False Positive = 42, Precision = 0.993, Recall = 0.999, F-Score = 0.996
Threshold >= 0.5: True Positive = 5975, False Positive = 0, Precision = 1.000, Recall = 0.999, F-Score = 1.000
Threshold >= 0.6: True Positive = 5936, False Positive = 0, Precision = 1.000, Recall = 0.993, F-Score = 0.996
Threshold >= 0.7: True Positive = 5900, False Positive = 0, Precision = 1.000, Recall = 0.987, F-Score = 0.993
Threshold >= 0.8: True Positive = 5894, False Positive = 0, Precision = 1.000, Recall = 0.986, F-Score = 0.993
Threshold >= 0.9: True Positive = 5886, False Positive = 0, Precision = 1.000, Recall = 0.984, F-Score = 0.992
```
```
Performance on the test data (20%)
```
```rust
Threshold >= 0.1: True Positive = 1506, False Positive = 445, Precision = 0.772, Recall = 0.980, F-Score = 0.864
Threshold >= 0.2: True Positive = 1497, False Positive = 298, Precision = 0.834, Recall = 0.974, F-Score = 0.899
Threshold >= 0.3: True Positive = 1490, False Positive = 256, Precision = 0.853, Recall = 0.969, F-Score = 0.908
Threshold >= 0.4: True Positive = 1456, False Positive = 97, Precision = 0.938, Recall = 0.947, F-Score = 0.942
Threshold >= 0.5: True Positive = 1450, False Positive = 88, Precision = 0.943, Recall = 0.943, F-Score = 0.943
Threshold >= 0.6: True Positive = 1448, False Positive = 85, Precision = 0.945, Recall = 0.942, F-Score = 0.943
Threshold >= 0.7: True Positive = 1387, False Positive = 40, Precision = 0.972, Recall = 0.902, F-Score = 0.936
Threshold >= 0.8: True Positive = 1379, False Positive = 38, Precision = 0.973, Recall = 0.897, F-Score = 0.934
Threshold >= 0.9: True Positive = 1371, False Positive = 38, Precision = 0.973, Recall = 0.892, F-Score = 0.931

```
</details>


# Evaluation

## Training Data
Trained and tested with the following datasets:
-  enronSpamSubset.csv
-  lingSpam.csv
-  completeSpamAssassin.csv
-  youtubeSpamCollection.csv
-  smsspamcollection.csv
-  governance_proposal_spam_likelihood.csv

```
total: 27.982
---------------
count spam: 9.012
count ham: 18.970
---------------

Failed to generate embeddings for all entries using Llama.cpp. However, embeddings were successfully generated for the following data: 

Number of Spam entries: 7237
Number of Ham entries: 13706
Total entries: 23310

```
<details>
<summary> <b>Expand to display the full dataset breakdown </b> </summary>

```
enronSpamSubset.csv
---------------
count spam: 5000
count ham: 5000

lingSpam.csv
---------------
count spam: 433
count ham: 2172

completeSpamAssassin.csv
---------------
count spam: 1560
count ham: 3952

youtubeSpamCollection.csv
---------------
count spam: 1005
count ham: 951
 
smsspamcollection.csv
---------------
count spam: 747
count ham: 4825

governance_proposal_spam_likelihood.csv
--------------- 
count spam: ?
count ham: ?
``` 
</details>

# 
This project is part of [CosmosRustBot](https://github.com/Philipp-Sc/cosmos-rust-bot), which provides Governance Proposal Notifications for Cosmos Blockchains. The goal is automatically detect fraudulent and deceitful proposals to prevent users falling for crypto scams. The current model is very effective in detecting fake governance proposals.

