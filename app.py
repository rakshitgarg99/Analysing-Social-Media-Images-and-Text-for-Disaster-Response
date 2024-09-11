from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
from nltk.corpus import stopwords
import nltk
import re

# Download stopwords dataset from NLTK
nltk.download('stopwords')

# Initialize Flask app and configure upload folder
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load image-based disaster prediction model
image_model = load_model('models/resnet_disaster_model.keras')

# Preprocessing function for image data
datagen = ImageDataGenerator(rescale=1./255)

# Function to preprocess input images before feeding to the model
def preprocess_image(img_path, target_size=(256, 256)):
    # Load and resize the image
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    # Expand the dimensions to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    # Rescale image using ImageDataGenerator
    img_array = datagen.flow(img_array, batch_size=1)[0]
    return img_array

# Load the text-based disaster prediction model
text_model = tf.keras.models.load_model('models/bert_disaster_model.h5')

# Load the tokenizer used for text preprocessing
with open('models/bert_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Set the maximum sequence length for text input
MAX_LEN = 100

# Text cleaning functions
def remove_url(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_html(text):
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub(r'', text)

def remove_punctuation(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

def remove_numbers(text):
    return re.sub(r'[0-9]+', '', text)

def remove_empty_spaces(text):
    return " ".join([word for word in text.split() if word.strip() != ""])

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word not in stop_words])

# Function to clean text before feeding to the model
def clean_text(text):
    text = text.lower()
    text = remove_url(text)
    text = remove_html(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_empty_spaces(text)
    text = remove_stopwords(text)
    return text

# Function to make predictions using the text model
def predict_disaster(text):
    # Convert text to sequence and pad to max length
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = text_model.predict(padded)[0][0]

    # Classify based on the prediction score (threshold = 0.5)
    is_disaster = prediction > 0.5
    confidence = float(prediction)

    result_text = "Disaster text detected" if is_disaster else "Non-disaster text detected"
    
    return {
        "result": result_text,
        "confidence": confidence
    }

# Main route to render the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image prediction requests
@app.route('/predict_image', methods=['POST'])
def predict_image():
    # Check if the image file is uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img_file = request.files['image']
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
    img_file.save(img_path)

    # Preprocess the uploaded image
    processed_image = preprocess_image(img_path)
    prediction = image_model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Define class labels
    class_labels = {
        0: 'Land Disaster',
        1: 'Fire Disaster',
        2: 'Non-Damage',
        3: 'Damaged Infrastructure',
        4: 'Water Disaster',
        5: 'Human Damage'
    }
    predicted_label = class_labels.get(predicted_class, 'Unknown')

    # Return the prediction result as JSON
    return jsonify({'predicted_class': predicted_label})

# Route to handle text prediction requests
@app.route('/predict_text', methods=['POST'])
def predict_text():
    # Get JSON data from the POST request
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Please provide the text for prediction."}), 400

    # Clean the input text
    text = clean_text(data['text'])

    # Perform the prediction
    result = predict_disaster(text)

    # Return the prediction result as JSON
    return jsonify({
        'predicted_class': result['result'],
        'confidence': result['confidence']
    })

# Run the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
