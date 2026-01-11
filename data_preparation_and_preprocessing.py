import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import faiss


df = pd.read_csv("data/client_hostname.csv")

cleaned_data = df[['client', 'hostname', 'alias_list', 'address_list']].copy()

client_encoder = LabelEncoder()
cleaned_data['client_encoded'] = client_encoder.fit_transform(cleaned_data['client'])

hostname_encoder = LabelEncoder()
cleaned_data['hostname_encoded'] = hostname_encoder.fit_transform(cleaned_data['hostname'])

alias_list_encoder = LabelEncoder()
cleaned_data['alias_list_encoded'] = alias_list_encoder.fit_transform(cleaned_data['alias_list'].astype(str))

address_list_encoder = LabelEncoder()
cleaned_data['address_list_encoded'] = address_list_encoder.fit_transform(cleaned_data['address_list'].astype(str))

vectorized_data = cleaned_data[['client_encoded', 'hostname_encoded', 'alias_list_encoded', 'address_list_encoded']].values
print("Vectorized Data Shape:", vectorized_data.shape)

vector_dim = vectorized_data.shape[1]  # Update dimension based on encoded columns
index = faiss.IndexFlatL2(vector_dim)

vectorized_data = np.array(vectorized_data, dtype=np.float32)
index.add(vectorized_data)
print("Number of vectors in index:", index.ntotal)


