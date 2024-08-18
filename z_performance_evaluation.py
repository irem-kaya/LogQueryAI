import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

# Test verilerini yükleyin
df_test = pd.read_csv("data/client_hostname.csv")

# Küçük bir test veri alt kümesi ile başlayın
test_sample_size = 100
df_test_sample = df_test.sample(n=test_sample_size, random_state=42)

# Test verilerinden soru-cevap çiftlerini oluşturun
def generate_question_answer_pairs(df):
    qa_pairs = []
    for _, row in df.iterrows():
        qa_pairs.append(f"Soru: {row['address_list']} IP adresinin hostname'i nedir?\nYanıt: {row['hostname']}")
        qa_pairs.append(f"Soru: {row['address_list']} IP adresinin alias listesi nedir?\nYanıt: {row['alias_list']}")
    return qa_pairs

test_data = generate_question_answer_pairs(df_test_sample)

# Modeli ve tokenizer'ı yükleyin
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)

# Modeli değerlendirme moduna alın
model.eval()

# Tahmin yapma
def generate_predictions(questions):
    inputs = tokenizer(questions, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs

def get_embeddings(outputs):
    # Model çıktılarından gömme vektörlerini al
    hidden_states = outputs.hidden_states
    embeddings = hidden_states[-1].mean(dim=1)
    return embeddings

def calculate_similarity(predictions, true_answers):
    # Get embeddings for predictions and true answers
    pred_outputs = generate_predictions(predictions)
    true_outputs = generate_predictions(true_answers)
    pred_embeddings = get_embeddings(pred_outputs)
    true_embeddings = get_embeddings(true_outputs)
    # Normalize embeddings
    pred_embeddings = torch.nn.functional.normalize(pred_embeddings, p=2, dim=1)
    true_embeddings = torch.nn.functional.normalize(true_embeddings, p=2, dim=1)
    # Calculate cosine similarity
    similarity_scores = []
    for pred_emb, true_emb in zip(pred_embeddings, true_embeddings):
        similarity = cosine_similarity(pred_emb.numpy().reshape(1, -1), true_emb.numpy().reshape(1, -1))
        similarity_scores.append(similarity[0][0])
    return np.mean(similarity_scores)

# Gerçek yanıtlar ve tahminler
true_answers = [pair.split('\n')[1].split('Yanıt: ')[1] for pair in test_data]
predictions = [pair.split('\n')[0] for pair in test_data]

# Sonuçlar ölçüldü.
# Ortalama benzerlik hesaplama
average_similarity = calculate_similarity(predictions, true_answers)
print(f"Ortalama Benzerlik: {average_similarity * 100:.2f}%")
