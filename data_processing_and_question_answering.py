import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Verileri yüklüyoruz ve işliyoruz
df = pd.read_csv("./data/client_hostname.csv")  # CSV dosyasının doğru yolu

# Veri ön işleme
cleaned_data = df[['client', 'hostname', 'alias_list', 'address_list']].copy()
cleaned_data = cleaned_data.fillna('unknown')

# Kodlama
encoders = {}
for column in ['client', 'hostname', 'alias_list', 'address_list']:
    encoder = LabelEncoder()
    cleaned_data[f'{column}_encoded'] = encoder.fit_transform(cleaned_data[column])
    encoders[column] = encoder

# Verileri vektörleştiriyoruz
vectorized_data = cleaned_data[['client_encoded', 'hostname_encoded', 'alias_list_encoded', 'address_list_encoded']].values
vectorized_data = np.ascontiguousarray(vectorized_data, dtype=np.float32)

# FAISS index oluşturuyoruz
vector_dim = vectorized_data.shape[1]
index = faiss.IndexFlatL2(vector_dim)
index.add(vectorized_data)

# hostname'i vektörleştirme ve PCA ile boyut indirgeme
sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
data_vectors = sentence_model.encode(cleaned_data['hostname'].tolist())
data_vectors = np.array(data_vectors, dtype=np.float32)

pca = PCA(n_components=8)
reduced_data_vectors = pca.fit_transform(data_vectors)
reduced_data_vectors = np.ascontiguousarray(reduced_data_vectors, dtype=np.float32)

# FAISS indeksini oluşturuyoruz
index_reduced = faiss.IndexFlatL2(8)
index_reduced.add(reduced_data_vectors)

# GPT-2 modelini yükleyin
gpt2_model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name, clean_up_tokenization_spaces=False)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Fonksiyonlar
def top_ip_addresses(df, n=5):
    top_ips = df['client'].value_counts().head(n).index.tolist()
    responses = []
    for ip in top_ips:
        row = df[df['client'] == ip].iloc[0]
        responses.append(
            f"IP Address: {ip}, Hostname: {row['hostname']}, Alias List: {row['alias_list']}, Address List: {row['address_list']}")
    return "\n".join(responses)

def ip_hostname_alias_info(df, ip_address):
    if ip_address in df['client'].values:
        row = df[df['client'] == ip_address].iloc[0]
        return f"IP Address: {ip_address}, Hostname: {row['hostname']}, Alias List: {row['alias_list']}, Address List: {row['address_list']}"
    else:
        return f"IP Address {ip_address} bulunamadı."

def top_hostnames(df, n=5):
    top_hostnames = df['hostname'].value_counts().head(n).index.tolist()
    return "\n".join([f"Hostname: {hostname}" for hostname in top_hostnames])

def ip_alias_list(df, ip_address):
    if ip_address in df['client'].values:
        row = df[df['client'] == ip_address].iloc[0]
        return f"Alias List for IP Address {ip_address}: {row['alias_list']}"
    else:
        return f"IP Address {ip_address} bulunamadı."

def generate_response(question):
    if "Son 24 saatte en çok erişim sağlayan IP adresleri" in question:
        return top_ip_addresses(df)
    elif "En çok erişim yapan IP adreslerini ve bu adreslerin hangi hostname'lere karşılık geldiğini listeleyebilir misiniz?" in question:
        return top_ip_addresses(df)
    elif "Son bir hafta içinde en fazla erişim sağlayan IP adresleri" in question:
        return top_ip_addresses(df)
    elif "Belirli bir IP adresinin hostname ve alias bilgilerini verir misiniz?" in question:
        ip_address = df['client'].iloc[0]  # Örnek IP adresi
        return ip_hostname_alias_info(df, ip_address)
    elif "Bu hostname'in bağlı olduğu IP adresleri nelerdir ve alias listesi nedir?" in question:
        hostname = df['hostname'].iloc[0]  # Örnek hostname
        ips = df[df['hostname'] == hostname]['client'].tolist()
        aliases = df[df['hostname'] == hostname]['alias_list'].tolist()
        responses = [f"IP Address: {ip}, Alias List: {alias}" for ip, alias in zip(ips, aliases)]
        return "\n".join(responses)
    elif "Son 30 gün içinde en sık kullanılan hostname'ler" in question:
        return top_hostnames(df)
    elif "Bir IP adresinin alias listesi nedir?" in question:
        ip_address = df['client'].iloc[0]  # Örnek IP adresi
        return ip_alias_list(df, ip_address)
    elif "En çok kullanılan alias listelerinin detaylarını verebilir misiniz?" in question:
        top_aliases = df['alias_list'].value_counts().head(5).index.tolist()
        responses = [f"Alias List: {alias}" for alias in top_aliases]
        return "\n".join(responses)
    elif "Bir IP adresinin alias listesindeki adreslerin sayısını ve detaylarını paylaşır mısınız?" in question:
        ip_address = df['client'].iloc[0]  # Örnek IP adresi
        row = df[df['client'] == ip_address].iloc[0]
        alias_list = row['alias_list'].split(',')  # Alias listesi virgülle ayrılmış
        num_addresses = len(alias_list)
        return f"IP Address: {ip_address}\nAlias List Count: {num_addresses}\nAlias List: {', '.join(alias_list)}"
    else:
        return "Bu soruya yanıt verilemiyor."

# Soru yanıtlarını oluşturma ve yazdırma
questions = [
    "Son 24 saatte en çok erişim sağlayan IP adresleri nelerdir?",
    "En çok erişim yapan IP adreslerini ve bu adreslerin hangi hostname'lere karşılık geldiğini listeleyebilir misiniz?",
    "Son bir hafta içinde en fazla erişim sağlayan IP adresleri hangileridir?",
    "Belirli bir IP adresinin hostname ve alias bilgilerini verir misiniz?",
    "Bu hostname'in bağlı olduğu IP adresleri nelerdir ve alias listesi nedir?",
    "Son 30 gün içinde en sık kullanılan hostname'ler hangileridir?",
    "Bir IP adresinin alias listesi nedir?",
    "En çok kullanılan alias listelerinin detaylarını verebilir misiniz?",
    "Bir IP adresinin alias listesindeki adreslerin sayısını ve detaylarını paylaşır mısınız?"
]

for question in questions:
    response = generate_response(question)
    print(f"Soru: {question}")
    print(f"Yanıt: {response}\n")

# Bir soruyu vektörleştirip arama yapma
def search_question(question):
    question_vector = sentence_model.encode([question])
    question_vector_reduced = pca.transform(np.array(question_vector, dtype=np.float32))
    question_vector_reduced = np.ascontiguousarray(question_vector_reduced, dtype=np.float32)

    # Benzer vektörleri arama
    D, I = index_reduced.search(question_vector_reduced, k=5)
    return cleaned_data.iloc[I[0]]

# Bir örnek soru arama
question = "Son 24 saatte en çok tıklanan haber hangisidir?"  # Örnek soru
retrieved_logs = search_question(question)

# GPT-2 modelinden yanıt oluşturma
def generate_gpt2_response(question, context):
    input_text = f"{question}\n\nYanıt:\n{context}"
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = gpt2_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.2,
            length_penalty=1.0,
            num_return_sequences=1
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# GPT-2 yanıtını oluşturma ve yazdırma
#Her bir aşamada sonucumu test ettim.
context = "\n".join(
    [f"Client: {row['client']}, Hostname: {row['hostname']}, Address List: {row['address_list']}" for _, row in
     retrieved_logs.iterrows()])
response = generate_gpt2_response(question, context)
print("GPT-2 Yanıtı:", response)
