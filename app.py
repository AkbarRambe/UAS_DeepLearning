from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Inisialisasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model RNN untuk ekspresi wajah
def create_rnn_model():
    model = Sequential([
        SimpleRNN(128, input_shape=(48, 48), activation='relu', return_sequences=True),
        Dropout(0.3),
        SimpleRNN(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(5, activation='softmax')  # 5 kelas ekspresi wajah
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Dummy model (belum dilatih)
model = create_rnn_model()

# Halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint untuk upload dan prediksi gambar
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Proses gambar
        img = load_img(filepath, color_mode='grayscale', target_size=(48, 48))
        img_array = img_to_array(img) / 255.0  # Normalisasi ke [0, 1]
        img_array = img_array.reshape(1, 48, 48)  # RNN membutuhkan input (batch, timesteps, features)

        # Prediksi menggunakan model RNN
        prediction = model.predict(img_array)
        classes = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral']
        predicted_class = classes[np.argmax(prediction)]

        return render_template('result.html', image_url=filepath, prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)