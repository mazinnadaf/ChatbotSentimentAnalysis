## Mazin Nadaf 2/8/2025
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report

# LSTM Neural Network
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return self.softmax(out)

def train_and_save_model(df, model_path='sentiment_model.pth'):
    def tokenize_and_build_vocab(texts):
        tokens = [word_tokenize(sentence.lower()) for sentence in texts]
        flattened = [word for sentence in tokens for word in sentence]
        vocab = {word: idx + 1 for idx, word in enumerate(set(flattened))}
        return tokens, vocab

    def encode_text(tokens, vocab):
        return [[vocab[token] for token in sentence if token in vocab] for sentence in tokens]

    texts = df["User Input"].tolist()
    tokens, vocab = tokenize_and_build_vocab(texts)

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["sentiment"])

    # Pad sequences
    sequences = encode_text(tokens, vocab)
    padded_sequences = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

    # Dataset and DataLoader
    class SentimentDataset(Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return self.inputs[idx], self.labels[idx]

    train_dataset = SentimentDataset(X_train, torch.tensor(y_train))
    test_dataset = SentimentDataset(X_test, torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model initialization
    vocab_size = len(vocab) + 1
    embed_dim = 128
    hidden_dim = 64
    output_dim = len(label_encoder.classes_)
    model = SentimentLSTM(vocab_size, embed_dim, hidden_dim, output_dim)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1000

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'label_encoder': label_encoder,
    }, model_path)
    print(f"Model saved at {model_path}")

    # Evaluate model
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions = torch.argmax(outputs, dim=1)
            y_pred.extend(predictions.tolist())
            y_true.extend(y_batch.tolist())

    print("Unique classes in y_true:", set(y_true))
    print("Unique classes in y_pred:", set(y_pred))
    print("Label encoder classes:", label_encoder.classes_)

    # Ensure correct label set
    unique_labels = sorted(set(y_true) | set(y_pred))
    print("Adjusted unique labels for classification report:", unique_labels)

    # Generate classification report with correct labels
    print("Classification Report:\n", classification_report(y_true, y_pred, labels=unique_labels))
