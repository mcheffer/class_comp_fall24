import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
from sklearn.metrics import accuracy_score, f1_score

# Download necessary NLTK data
nltk.download('punkt')

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', additional_feature_dim=3, num_labels=2):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        
        combined_dim = self.bert.config.hidden_size + additional_feature_dim
        self.classifier = nn.Linear(combined_dim, num_labels)

    def forward(self, input_ids, attention_mask, additional_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # CLS token representation
        pooled_output = self.dropout(pooled_output)
        combined_output = torch.cat((pooled_output, additional_features), dim=1)
        logits = self.classifier(combined_output)
        return logits

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, additional_features, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.additional_features = additional_features
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx]) if self.labels is not None else None
        additional_feature = torch.tensor(self.additional_features.iloc[idx].values, dtype=torch.float)

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'additional_features': additional_feature
        }
        if label is not None:
            item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

    def __len__(self):
        return len(self.texts)

def calculate_word_frequencies(df, label):
    filtered_texts = df[df['LABEL'] == label]['TEXT']
    all_words = filtered_texts.apply(word_tokenize).sum()
    return Counter(all_words)

def extract_text_features(text, top_author_words, top_non_author_words):
    if not isinstance(text, str) or not text.strip():
        return pd.Series({'author_word_count': 0, 'non_author_word_count': 0, 'text_length': 0})
    words = word_tokenize(text)
    author_word_count = sum(1 for word in words if word in top_author_words)
    non_author_word_count = sum(1 for word in words if word in top_non_author_words)
    text_length = len(words)
    return pd.Series({
        'author_word_count': author_word_count,
        'non_author_word_count': non_author_word_count,
        'text_length': text_length
    })

train_df = pd.read_csv('train.csv')
train_df = train_df.dropna(subset=['TEXT'])


author_word_freq = calculate_word_frequencies(train_df, label=1)
non_author_word_freq = calculate_word_frequencies(train_df, label=0)

top_author_words = {word for word, _ in author_word_freq.most_common(50)}
top_non_author_words = {word for word, _ in non_author_word_freq.most_common(50)}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 512
batch_size = 16

train_df[['author_word_count', 'non_author_word_count', 'text_length']] = train_df['TEXT'].apply(
    extract_text_features, args=(top_author_words, top_non_author_words)
)
texts = train_df['TEXT']
labels = train_df['LABEL']
additional_features = train_df[['author_word_count', 'non_author_word_count', 'text_length']]
train_dataset = TextDataset(texts, labels, additional_features, tokenizer, max_len)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier().to(device)

def train_model(model, train_loader, device, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    training_logs = []

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            additional_features = batch['additional_features'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask, additional_features)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        training_logs.append(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    with open('training_logs.txt', 'w') as f:
        f.writelines("\n".join(training_logs))

    return model

print("train_model")
trained_model = train_model(model, train_loader, device, epochs=1)
torch.save(trained_model.state_dict(), 'bert_author_model.pth')

test_df = pd.read_csv('test.csv')
test_df = test_df.dropna(subset=['TEXT'])
test_df[['author_word_count', 'non_author_word_count', 'text_length']] = test_df['TEXT'].apply(
    extract_text_features, args=(top_author_words, top_non_author_words)
)

test_texts = test_df['TEXT']
test_additional_features = test_df[['author_word_count', 'non_author_word_count', 'text_length']]
test_dataset = TextDataset(test_texts, None, test_additional_features, tokenizer, max_len)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            additional_features = batch['additional_features'].to(device)

            logits = model(input_ids, attention_mask, additional_features)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions

print("evaluating_model")
predictions = evaluate_model(trained_model, test_loader, device)

# Save predictions
output = pd.DataFrame({'ID': test_df['ID'], 'LABEL': predictions})
output.to_csv('submission.csv', index=False)