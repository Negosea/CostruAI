# src/models/train_model.py

from sklearn.ensemble import RandomForestClassifier

def train_model(data):
    model = RandomForestClassifier()
    model.fit(data['features'], data['labels'])
    return model
