# import fungsi dari transformer
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from nltk.tokenize import TweetTokenizer

# import library pytorch dan library lain yg diperlukan
import os, sys
sys.path.append('../')
os.chdir('../')
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# import fungsi dari package utils
from google.colab import drive
drive.mount('/content/drive')
sys.path.append('/content/drive/MyDrive/ColabNotebooks/utils')
from forward_fn import forward_sequence_classification
from metrics import document_sentiment_metrics_fn
from data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader

# common functions
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def count_param(module, trainable=False):
    if trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)

# Set random seed
set_seed(26092020)

# Muat tokenizer dan konfigurasi
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
config = BertConfig.from_pretrained('indobenchmark/indobert-base-p1')
config.num_labels = DocumentSentimentDataset.NUM_LABELS

# Instansiasi model pre-trained
model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p1', config=config)

train_dataset_path = '/content/drive/MyDrive/ColabNotebooks/dataset/train_preprocess.tsv'
valid_dataset_path = '/content/drive/MyDrive/ColabNotebooks/dataset/valid_preprocess.tsv'

train_dataset = DocumentSentimentDataset(train_dataset_path, tokenizer, lowercase=True)
valid_dataset = DocumentSentimentDataset(valid_dataset_path, tokenizer, lowercase=True)

train_loader = DocumentSentimentDataLoader(dataset=train_dataset, max_seq_len=512,
                                           batch_size=32, num_workers=2, shuffle=True)
valid_loader = DocumentSentimentDataLoader(dataset=valid_dataset, max_seq_len=512,
                                           batch_size=32, num_workers=2, shuffle=False)

w2i, i2w = DocumentSentimentDataset.LABEL2INDEX, DocumentSentimentDataset.INDEX2LABEL

# Set optimizer, lr, epoch
optimizer = optim.Adam(model.parameters(), lr=3e-6)
model = model.cuda()
n_epochs = 5

# Siapkan array untuk menampung acc dan loss selama train
train_losses = []
valid_losses = []
train_acc = []
valid_acc = []

for epoch in range(n_epochs):
    model.train()
    torch.set_grad_enabled(True)
    total_train_loss = 0
    list_hyp, list_label = [], []

    train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
    for i, batch_data in enumerate(train_pbar):
        # Forward model
        loss, batch_hyp, batch_label = forward_sequence_classification(model, batch_data[:-1], i2w=i2w, device='cuda')
        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        total_train_loss = total_train_loss + train_loss
        # Calculate metrics
        list_hyp += batch_hyp
        list_label += batch_label
        train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
            total_train_loss/(i+1), get_lr(optimizer)))
    # Calculate train metric
    metrics = document_sentiment_metrics_fn(list_hyp, list_label)
    print("TRAIN LOSS:{:.4f} {} LR:{:.8f}".format(
        total_train_loss/(i+1), metrics_to_string(metrics), get_lr(optimizer)))
    # Append the training loss for this epoch
    train_losses.append(total_train_loss / (i + 1))
    train_acc.append(metrics["ACC"])

    # Evaluate on validation
    model.eval()
    torch.set_grad_enabled(False)
    total_loss, total_correct, total_labels = 0, 0, 0
    list_hyp, list_label = [], []

    val_pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
    for i, batch_data in enumerate(val_pbar):
        batch_seq = batch_data[-1]
        loss, batch_hyp, batch_label = forward_sequence_classification(model,
                                                    batch_data[:-1], i2w=i2w, device='cuda')
        # Calculate total loss
        valid_loss = loss.item()
        total_loss = total_loss + valid_loss

        # Calculate evaluation metrics
        list_hyp += batch_hyp
        list_label += batch_label
        metrics = document_sentiment_metrics_fn(list_hyp, list_label)
        val_pbar.set_description("(Epoch {}) VALID LOSS:{:.4f} {}".
                                 format((epoch+1),total_loss/(i+1), metrics_to_string(metrics)))

    metrics = document_sentiment_metrics_fn(list_hyp, list_label)
    print("VALID LOSS:{:.4f} {}".format(
        total_loss/(i+1), metrics_to_string(metrics)))
    # Append the training loss for this epoch
    valid_losses.append(total_loss / (i + 1))
    valid_acc.append(metrics["ACC"])

# Save prediction
df = pd.DataFrame({'label':list_hyp}).reset_index()
df.to_csv('pred.txt', index=False)

# uji coba manual
def testModel(text):
    subwords = tokenizer.encode(text)
    subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

    logits = model(subwords)[0]
    label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

    print(f'Text: {text} | Label : {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)')

text = input()
testModel(text)