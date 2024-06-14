import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Carregar dados
data = pd.read_csv('dataset.csv')

# Pré-processamento dos dados
label_encoder = LabelEncoder()
data['activatePump'] = label_encoder.fit_transform(data['activatePump'])

X = data.drop('activatePump', axis=1)
y = data['activatePump']

# Treinar o modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Salvar o modelo e o label encoder
joblib.dump(model, 'model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Modelo treinado e salvo com sucesso!")

