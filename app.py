import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

app = Flask(__name__)

# Визначаємо архітектуру моделі
def create_model():
    inputs = layers.Input(shape=(4,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(70, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# Витягуємо ваги з crop_model.keras
print("1. Витягуємо ваги з crop_model.keras...")

import zipfile
import h5py

weights_dict = {}

with zipfile.ZipFile('crop_model.keras', 'r') as z:
    weights_files = [f for f in z.namelist() if f.endswith('.weights.h5')]
    if not weights_files:
        print("Файл з вагами не знайдено")
        exit(1)
    
    print(f"   Знайдено файл ваг: {weights_files[0]}")
    
    with z.open(weights_files[0]) as f:
        with h5py.File(f, 'r') as hf:
            def collect_weights(name, obj):
                if isinstance(obj, h5py.Dataset) and 'vars' in name:
                    weights_dict[name] = obj[()]
                    print(f"   Завантажено: {name} -> {obj.shape}")
            
            hf.visititems(collect_weights)

print(f"\n Всього завантажено {len(weights_dict)} тензорів ваг")


# Збираємо ваги для нової моделі
print("\n2. Збираємо ваги для нової моделі...")

# Ваги мають йти в такому порядку:
# dense/kernel, dense/bias, dense_1/kernel, dense_1/bias, dense_2/kernel, dense_2/bias, dense_3/kernel, dense_3/bias

layer_weights = []

# Шар 1 (dense): kernel (4,128), bias (128,)
layer_weights.append(weights_dict['layers/dense/vars/0'])  # kernel
layer_weights.append(weights_dict['layers/dense/vars/1'])  # bias

# Шар 2 (dense_1): kernel (128,64), bias (64,)
layer_weights.append(weights_dict['layers/dense_1/vars/0'])  # kernel
layer_weights.append(weights_dict['layers/dense_1/vars/1'])  # bias

# Шар 3 (dense_2): kernel (64,32), bias (32,)
layer_weights.append(weights_dict['layers/dense_2/vars/0'])  # kernel
layer_weights.append(weights_dict['layers/dense_2/vars/1'])  # bias

# Шар 4 (dense_3): kernel (32,70), bias (70,)
layer_weights.append(weights_dict['layers/dense_3/vars/0'])  # kernel
layer_weights.append(weights_dict['layers/dense_3/vars/1'])  # bias

print(f"Підготовлено {len(layer_weights)} тензорів ваг")


# 4. Створюємо модель і встановлюємо модель
print("\n3. Створення нової моделі...")
model = create_model()
print("Нову модель створено")

print("\n4. Встановлення ваг...")
model.set_weights(layer_weights)
print("Ваги успішно встановлено")

# Перевірка, що модель працює
test_input = np.array([[23.0, 70.0, 6.5, 200.0]])
test_output = model.predict(test_input)
print(f"Тестове передбачення працює: форма виходу {test_output.shape}")

# Завантажуємо додаткові файли
print("\n5. Завантаження мапінгу культур та параметрів масштабування...")

with open('crop_label_mapping.json', 'r', encoding='utf-8') as f:
    crop_mapping = json.load(f)

with open('scaler_params.json', 'r', encoding='utf-8') as f:
    scaler_params = json.load(f)

MEANS = scaler_params['mean']
SCALES = scaler_params['scale']

print(f" Мапінг культур завантажено (культур: {len(crop_mapping)})")
print(" Параметри масштабування завантажено")
print(" СЕРВЕР ГОТОВИЙ ДО РОБОТИ")
print("="*50 + "\n")


# Функція масштабування 
def scale_input(temperature, humidity, ph, rainfall):
    return [
        (temperature - MEANS[0]) / SCALES[0],
        (humidity - MEANS[1]) / SCALES[1],
        (ph - MEANS[2]) / SCALES[2],
        (rainfall - MEANS[3]) / SCALES[3]
    ]

# Маршрути FLASK
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        
        scaled = scale_input(temperature, humidity, ph, rainfall)
        input_tensor = np.array([scaled])
        
        predictions = model.predict(input_tensor)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)
        
        crop_name = crop_mapping.get(str(predicted_class), "Невідома культура")
        
        return jsonify({
            'success': True,
            'crop': crop_name,
            'confidence': round(confidence, 2),
            'class_id': predicted_class
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Запуск
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)