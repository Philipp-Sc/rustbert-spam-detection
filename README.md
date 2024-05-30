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

Git clone and train the required models:

```bash
cargo run --release train_and_test_text_embedding_knn_regressor
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
[0.16443461, 0.0062025306, 0.6938212, 0.0014256272, 0.9994333, 0.043457787]


[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
```

# Architecture

## Features

### KNN Regressor on Text Embeddings (llama.cpp) ⭐
Single fraud likelihood prediction based on text embeddings.
Model: [uae-large-v1_fp32.gguf](https://huggingface.co/ChristianAzinn/uae-large-v1-gguf)

<details>
<summary> <b>Expand to display the full evaluation (F-Score = 0.948) </b> </summary>

```
Performance on the training data (80%)
```
```rust
Threshold >= 0.1: True Positive = 7180, False Positive = 890, Precision = 0.890, Recall = 1.000, F-Score = 0.942
Threshold >= 0.2: True Positive = 7180, False Positive = 222, Precision = 0.970, Recall = 1.000, F-Score = 0.985
Threshold >= 0.3: True Positive = 7175, False Positive = 143, Precision = 0.980, Recall = 0.999, F-Score = 0.990
Threshold >= 0.4: True Positive = 7166, False Positive = 65, Precision = 0.991, Recall = 0.998, F-Score = 0.995
Threshold >= 0.5: True Positive = 7166, False Positive = 0, Precision = 1.000, Recall = 0.998, F-Score = 0.999
Threshold >= 0.6: True Positive = 7114, False Positive = 0, Precision = 1.000, Recall = 0.991, F-Score = 0.995
Threshold >= 0.7: True Positive = 7072, False Positive = 0, Precision = 1.000, Recall = 0.985, F-Score = 0.992
Threshold >= 0.8: True Positive = 7063, False Positive = 0, Precision = 1.000, Recall = 0.984, F-Score = 0.992
Threshold >= 0.9: True Positive = 7056, False Positive = 0, Precision = 1.000, Recall = 0.983, F-Score = 0.991
```
```
Performance on the test data (20%)
```
```rust
Threshold >= 0.1: True Positive = 1802, False Positive = 539, Precision = 0.770, Recall = 0.984, F-Score = 0.864
Threshold >= 0.2: True Positive = 1798, False Positive = 395, Precision = 0.820, Recall = 0.981, F-Score = 0.893
Threshold >= 0.3: True Positive = 1790, False Positive = 317, Precision = 0.850, Recall = 0.977, F-Score = 0.909
Threshold >= 0.4: True Positive = 1742, False Positive = 123, Precision = 0.934, Recall = 0.951, F-Score = 0.942
Threshold >= 0.5: True Positive = 1739, False Positive = 103, Precision = 0.944, Recall = 0.949, F-Score = 0.947
Threshold >= 0.6: True Positive = 1736, False Positive = 95, Precision = 0.948, Recall = 0.948, F-Score = 0.948
Threshold >= 0.7: True Positive = 1658, False Positive = 49, Precision = 0.971, Recall = 0.905, F-Score = 0.937
Threshold >= 0.8: True Positive = 1645, False Positive = 47, Precision = 0.972, Recall = 0.898, F-Score = 0.934
Threshold >= 0.9: True Positive = 1640, False Positive = 46, Precision = 0.973, Recall = 0.895, F-Score = 0.932
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

