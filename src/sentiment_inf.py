import torch
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from senty_mdl_inf_svd_mdl import SentimentLSTM

class SentimentInference:
    def __init__(self, model_path='sentiment_model.pth'):
        # Load model and artifacts
        checkpoint = torch.load(model_path)
        self.vocab = checkpoint['vocab']
        self.label_encoder = checkpoint['label_encoder']
        vocab_size = len(self.vocab) + 1
        embed_dim = 128
        hidden_dim = 64
        output_dim = len(self.label_encoder.classes_)

        # Initialize and load the model
        self.model = SentimentLSTM(vocab_size, embed_dim, hidden_dim, output_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, sentences):
        def encode_text(tokens, vocab):
            return [[vocab[token] for token in sentence if token in vocab] for sentence in tokens]

        tokens = [word_tokenize(sentence.lower()) for sentence in sentences]
        sequences = encode_text(tokens, self.vocab)
        padded_sequences = pad_sequence(
            [torch.tensor(seq) for seq in sequences], batch_first=True
        )

        with torch.no_grad():
            outputs = self.model(padded_sequences)
            predictions = torch.argmax(outputs, dim=1)
            predicted_labels = self.label_encoder.inverse_transform(predictions.numpy())
        return predicted_labels