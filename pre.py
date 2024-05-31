import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Load saved embeddings and labels
embeddings = np.load('saved_embeddings.npy')
labels = np.load('labels.npy')

# Encode string labels into integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Train a K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_classifier.fit(embeddings, encoded_labels)

def identify_face(face_embedding):
    # Use the trained KNN classifier to predict
    predicted_label = knn_classifier.predict(face_embedding.reshape(1, -1))
    
    # Decode the predicted label back to string
    predicted_name = label_encoder.inverse_transform(predicted_label)[0]
    
    return predicted_name
