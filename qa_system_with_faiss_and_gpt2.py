import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# Adım 1: Veriyi yüklüyoruz ve datayı temizliyoruz
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    cleaned_data = df[['client', 'hostname', 'alias_list', 'address_list']].copy()
    cleaned_data = cleaned_data.fillna('unknown').astype(str)

    encoders = {
        'client': LabelEncoder(),
        'hostname': LabelEncoder(),
        'alias_list': LabelEncoder(),
        'address_list': LabelEncoder()
    }

    for column in encoders:
        cleaned_data[f"{column}_encoded"] = encoders[column].fit_transform(cleaned_data[column])

    return cleaned_data, encoders


# Adım 2: Verileri vektörlüyoruz ve PCA ile boyutları indiriyoruz
def create_faiss_index(cleaned_data):
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    data_vectors = model.encode(cleaned_data['hostname'].tolist())
    data_vectors = np.array(data_vectors, dtype=np.float32)

    pca = PCA(n_components=8)
    reduced_data_vectors = pca.fit_transform(data_vectors)
    reduced_data_vectors = np.ascontiguousarray(reduced_data_vectors, dtype=np.float32)

    index = faiss.IndexFlatL2(reduced_data_vectors.shape[1])
    index.add(reduced_data_vectors)

    return index, pca


# Soruya göre yanıt oluşturma
def generate_response(question, cleaned_data, index, pca, tokenizer, gpt2_model):
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    question_vector = model.encode(question)
    question_vector_reduced = pca.transform(np.array([question_vector], dtype=np.float32))
    question_vector_reduced = np.ascontiguousarray(question_vector_reduced, dtype=np.float32)

    D, I = index.search(question_vector_reduced, k=5)
    retrieved_logs = cleaned_data.iloc[I[0]]

    if "en çok" in question or "most" in question:
        # En çok tıklanan hostname'i bul
        most_clicked = retrieved_logs['hostname'].mode()[0]
        response = f"En çok tıklanan hostname: {most_clicked}"
    else:
        # Sıralı şekilde yanıt oluştur
        context = " ".join(
            [f"Client: {row['client']}, Hostname: {row['hostname']}, Address List: {row['address_list']}" for _, row in
             retrieved_logs.iterrows()]
        )
        input_text = f"Soru: {question}\nYanıt: {context}"
        inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = gpt2_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=550,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.2,
            length_penalty=1.0,
            num_return_sequences=1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = " ".join(response.split()[:550])

    return response

#Bu aşamada da yine kullanıcıdan soru alıp sonrasında benzer sorular ile ilişki kurarak cevaplar ürettim.
def main():
    filepath = "data/client_hostname.csv"

    cleaned_data, encoders = load_and_preprocess_data(filepath)
    index, pca = create_faiss_index(cleaned_data)

    gpt2_model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    question = input("Lütfen sorunuzu giriniz: ")
    response = generate_response(question, cleaned_data, index, pca, tokenizer, gpt2_model)
    print(response)


if __name__ == "__main__":
    main()
