import json
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기
def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts, labels = [], []
    for label_str, examples in data.items():
        for example in examples:
            texts.append(example["text"])
            labels.append(1 if label_str == "긍정" else 0)
    return texts, labels

# 2. 학습/검증 데이터셋 분할
def split_data(texts, labels, test_size=0.1, random_state=42):
    return train_test_split(texts, labels, test_size=test_size, random_state=random_state, stratify=labels)

# 3. 토크나이저 로드 및 토큰화
def tokenize_texts(text_list, tokenizer, max_length=256):
    return tokenizer(text_list, padding="max_length", truncation=True, max_length=max_length)

# 4. PyTorch 데이터셋 클래스 정의
class NewsSentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

# 5. 모델 로드
def load_model(model_name):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 6. 클래스 불균형을 반영한 가중치 계산
def compute_class_weights(labels):
    train_counts = Counter(labels)
    total_train = len(labels)
    num_classes = 2
    weights = [total_train / (num_classes * train_counts[i]) for i in range(num_classes)]
    class_weights = torch.tensor(weights, dtype=torch.float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return class_weights.to(device)

# 7. 가중치 기반 Trainer 정의
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 8. TrainingArguments 설정
def get_training_args(output_dir="beaming/robert_about"):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_dir=output_dir,
        logging_steps=10,
        save_total_limit=2,
        report_to=[]
    )

# 9. 실행 함수
def train_sentiment_model(data_path, model_name="zeropepsi/robert-ko-news-sentiment"):
    texts, labels = load_data(data_path)
    train_texts, val_texts, train_labels, val_labels = split_data(texts, labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenize_texts(train_texts, tokenizer)
    val_encodings = tokenize_texts(val_texts, tokenizer)
    train_dataset = NewsSentimentDataset(train_encodings, train_labels)
    val_dataset = NewsSentimentDataset(val_encodings, val_labels)
    model = load_model(model_name)
    class_weights = compute_class_weights(train_labels)
    training_args = get_training_args()
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        class_weights=class_weights
    )
    trainer.train()
    return model, trainer

# 실행 예시
if __name__ == "__main__":
    # model, trainer = train_sentiment_model("data/train_over09.json")