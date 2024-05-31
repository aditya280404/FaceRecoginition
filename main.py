from flask import Flask, render_template, request, Response
import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

app = Flask(__name__)

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load saved embeddings and labels
saved_embeddings = np.load('saved_embeddings.npy')
labels = np.load('labels.npy')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('/Users/aditya/finalMlPro/labels.npy')

def extract_faces(img):
    try:
        # Detect faces using MTCNN
        faces, _ = mtcnn.detect(img)
        return faces
    except Exception as e:
        print("Error in extract_faces:", e)
        return []





def train_model():
    faces = []
    labels = []
    userlist = os.listdir('/Users/aditya/finalMlPro/known_faces')
    for user in userlist:
        for imgname in os.listdir(f'/Users/aditya/finalMlPro/known_faces/{user}'):
            img = cv2.imread(f'/Users/aditya/finalMlPro/known_faces/{user}/{imgname}')
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            resized_face = cv2.resize(img_rgb, (160, 160))  # Resize face for InceptionResnetV1
            faces.append(resized_face)
            labels.append(user)
    if faces:  # Check if faces is not empty
        faces = np.array(faces)
        labels = np.array(labels)
        
        # Encode string labels into integers
        labels_encoded = label_encoder.fit_transform(labels)
        
        # Extract embeddings using InceptionResnetV1
        embeddings = []
        for face in faces:
            embedding = resnet(torch.tensor(face).unsqueeze(0).permute(0, 3, 1, 2).float()).detach().numpy()
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        
        # Save embeddings and labels
        np.save('saved_embeddings.npy', embeddings)
        np.save('labels.npy', labels_encoded)
        
    else:
        print("No faces found for training.")

def predict_identity(frame):
    # Preprocess frame using MTCNN to detect faces
    faces, _ = mtcnn.detect(frame)
    
    if faces is None:
        return "No face detected"
    
    print("Detected faces:", faces)  # Add logging statement
    
    # Filter out invalid elements from faces array
    valid_faces = [face for face in faces if isinstance(face, np.ndarray)]
    
    print("Valid faces:", valid_faces)  # Add logging statement
    
    if not valid_faces:
        return "No valid faces detected"
    
    # Extract the first valid face
    valid_face = valid_faces[0]
    
    # Convert the coordinates to integers
    x, y, w, h = [int(coord) for coord in valid_face]
    
    # Extract the face region from the frame
    face_region = frame[y:y+h, x:x+w]
    
    if face_region.size == 0:
        return "Empty face region"
    
    # Resize the face region to match the input size expected by the model
    resized_face = cv2.resize(face_region, (160, 160))
    
    # Convert the resized face to a PyTorch tensor and preprocess it
    face_tensor = torch.tensor(resized_face).unsqueeze(0).permute(0, 3, 1, 2).float()
    
    # Extract embeddings using InceptionResnetV1
    embeddings = resnet(face_tensor)
    
    # Calculate cosine similarity between embeddings
    distances = np.linalg.norm(embeddings.detach().numpy()[:, np.newaxis] - saved_embeddings, axis=2)
    min_distance_idx = np.argmin(distances, axis=1)
    min_distances = distances[np.arange(len(distances)), min_distance_idx]
    
    # Check if the minimum distance is below the threshold
    confidence_threshold = 0.3
    if np.min(min_distances) < confidence_threshold:
        predicted_class_index = min_distance_idx[np.argmin(min_distances)]
        predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
        confidence_score = 1 - min_distances[predicted_class_index]  # Calculate confidence score
        return predicted_label, confidence_score
    else:
        return "Unknown", 0  # Return confidence score of 0 for unknown faces

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform face recognition on the frame
            prediction = predict_identity(frame)

            # Draw prediction on the frame
            cv2.putText(frame, f'Prediction: {prediction}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    nimgs = int(request.form['nimgs'])

    userimagefolder = '/Users/aditya/finalMlPro/known_faces/' + newusername
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    
    # Create a folder in known_faces for the new user
    known_faces_folder = '/Users/aditya/finalMlPro/known_faces'
    if not os.path.isdir(known_faces_folder):
        os.makedirs(known_faces_folder)
    
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            faces = extract_faces(frame)
            if faces is not None:  # Check if faces were detected
                for (x, y, w, h) in faces:
                    x, y, w, h = int(x), int(y), int(w), int(h)  # Convert to integers
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                    cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                    if j % 5 == 0:
                        name = f'{newusername}_{i}.jpg'
                        cv2.imwrite(os.path.join(userimagefolder, name), frame[y:y+h, x:x+w])
                        i += 1
                    j += 1
                if i == nimgs:
                    break
            else:
                print("No faces detected in the frame.")
        else:
            print("Error capturing frame from webcam.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,port=5001)

