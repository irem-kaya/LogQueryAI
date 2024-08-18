
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset


df = pd.read_csv("data/client_hostname.csv")

# Küçük bir veri alt kümesi ile başlayın
sample_size = 200  # Eğitim için küçük bir alt küme (örneğin 1000 örnek)
df_sample = df.sample(n=sample_size, random_state=42)  # random_state, aynı alt kümenin her seferinde seçilmesini sağlar

# Soru-cevap çiftlerini oluşturuldu.
def generate_question_answer_pairs(df):
    qa_pairs = []
    for _, row in df.iterrows():
        qa_pairs.append(f"Soru: {row['address_list']} IP adresinin hostname'i nedir?\nYanıt: {row['hostname']}")
        qa_pairs.append(f"Soru: {row['address_list']} IP adresinin alias listesi nedir?\nYanıt: {row['alias_list']}")
    return qa_pairs

train_data = generate_question_answer_pairs(df_sample)

# Model ve tokenizer yüklendi.
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)

# Eğitim verilerini encode edildi.
train_encodings = tokenizer(train_data, return_tensors='pt', padding=True, truncation=True, max_length=512)

# Özel veri kümesi sınıfı tanımlandı.
class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone().detach()
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = CustomDataset(train_encodings)

# Eğitim ayarları yapıldı
from transformers import TrainingArguments

#batchleri artırarak hızlı çalışması sağlandı.
#epoch sayısı ile overfit olma durumu engellendi.
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,  # Batch boyutunu artırabilirsiniz
    per_device_eval_batch_size=8,   # Batch boyutunu artırabilirsiniz
    num_train_epochs=5,  # Epoch sayısını artırabilirsiniz
    learning_rate=3e-5,  # Öğrenme oranını ayarlayabilirsiniz
    warmup_steps=500,   # Warmup adımlarını ayarlayabilirsiniz
    weight_decay=0.01,  # Ağırlık azalmasını ayarlayabilirsiniz
    logging_dir="./logs",  # Log dizinini belirleyebilirsiniz
    logging_steps=10,  # Loglama sıklığını ayarlayabilirsiniz
)

# Trainer sınıfını oluşturun
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
#Gerekli testlerden sonra model eğitilmeye başlandı.
# Modeli eğitildi.
trainer.train()

# Eğitimli model ve tokenizer'ı kaydedildi.
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
