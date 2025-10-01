Protein Sequence Classification with Deep CNNs

This repository implements a deep learning pipeline to classify protein sequences into different labeled categories using convolutional neural networks (CNNs) on amino acid sequences.

🔗 Competition Link: InstaDeep Enzyme Classification Challenge – Zindi

🧪 Overview

Tokenize protein sequences using a custom ProteinTokenizer that maps amino acids to indices, with padding and unknown tokens.

Two CNN‐based architectures are implemented:

ProteinCNN: single convolutional layer per kernel size

DeepProteinCNN: two stacked convolutional blocks per kernel size with batch normalization

The model is trained with cross-entropy loss and evaluated on a held-out validation set.

After training, predictions on a test dataset are saved to a submission.csv file.

🚀 Setup & Installation

Clone the repository:
```bash
git clone <repo_url>
cd <repo_name>
```


Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Ensure you have PyTorch installed (with GPU support if available).

🧩 Tokenizer & Dataset

ProteinTokenizer

Maps the 20 standard amino acids to indices.

Includes special tokens: PAD, UNK.

Methods:
```
encode(seq): convert amino acid sequence (string) to list of token IDs

pad(ids, max_len): pad or truncate to max_len
```
ProteinDataset
```
Wraps a pandas DataFrame with columns:

SEQUENCE (string)

SEQUENCE_ID

LABEL (for training)

Produces tokenized & padded sequences and label indices (for training).
```
🧠 Model Architectures
🔹 ProteinCNN

Embedding layer
```
Multiple 1D convolutional filters (kernel sizes: e.g. 3, 5, 7)
```
Max pooling over sequence length

Fully connected output layer

🔹 DeepProteinCNN

Embedding layer

For each kernel size: Conv1D → BatchNorm → ReLU (stacked twice)

Max pooling

Dropout + fully connected output layer

Choose the architecture by instantiating either ProteinCNN or DeepProteinCNN.

🏋️ Training & Validation

Split Train.csv into train and validation (e.g. 90/10 stratified split).

Train for a set number of epochs (default = 10).

For each epoch:
```
Forward pass → compute loss → backward pass → optimizer step
``
Track training accuracy & loss

Evaluate on validation set (val accuracy & loss).
```
📊 Example training log:

Epoch 1 | Train Acc: 67.45% | Val Acc: 65.82%  
Epoch 2 | Train Acc: 72.13% | Val Acc: 69.50%  
...  
Epoch 10 | Train Acc: 84.60% | Val Acc: 81.20%  

📈 Results & Metrics

We evaluated the model on the validation set using standard classification metrics.

Accuracy: ~81% (DeepProteinCNN)

F1-score (macro avg): ~0.78

Confusion Matrix: Visualizes class distribution and misclassifications.

Example training vs validation loss curve:
```
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()
```

📊 Example Loss Curve:
(Insert your generated plot here if running in notebook)

🧾 Inference & Submission
```
Load Test.csv and wrap with ProteinDataset(is_test=True).
```
Run inference to get predicted label indices.

Map predictions back to labels using label_map.

Save submission file:
```
submission = pd.DataFrame({
    'SEQUENCE_ID': ids,
    'LABEL': [inv_map[i] for i in preds]
})
submission.to_csv("submission.csv", index=False)
```
```
🔧 Hyperparameters & Settings
Parameter	Default Value
max_len	512
embed_dim	128
kernel_sizes	[3, 5, 7]
num_filters	64
dropout	0.3
learning_rate	1e-3
batch_size	64
epochs	10
📈 Example Usage
```
Training & Validation
```
model = DeepProteinCNN(vocab_size=len(tokenizer.vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    # training loop ...
    # validation loop ...
    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")


Inference

test_ds = ProteinDataset(test_df, tokenizer, max_len, is_test=True)
test_loader = DataLoader(test_ds, batch_size=batch_size)

model.eval()
preds, ids = [], []
with torch.no_grad():
    for x, seq_ids in test_loader:
        x = x.to(device)
        out = model(x)
        _, pred = out.max(1)
        preds.extend(pred.cpu().numpy())
        ids.extend(seq_ids)

inv_map = {v: k for k, v in train_ds.label_map.items()}
submission = pd.DataFrame({
    'SEQUENCE_ID': ids,
    'LABEL': [inv_map[i] for i in preds]
})
submission.to_csv("submission.csv", index=False)
```
📝 License & Attribution

License: (e.g. MIT, Apache 2.0 — add your choice)

Dataset & competition task: Zindi Africa – InstaDeep Enzyme Classification Challenge
.

Frameworks: PyTorch
, scikit-learn
, pandas
.
