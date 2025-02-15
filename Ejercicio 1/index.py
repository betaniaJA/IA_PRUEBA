import tensorflow as tf
import numpy as np

# 1. Cargar un modelo preentrenado (MobileNetV2)

model = tf.keras.applications.MobileNet(weights='imagenet')
# 2. Definir la función de preprocesamiento
def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))  # Redimensionar
    img_array = tf.keras.preprocessing.image.img_to_array(img)                    # Convertir a array
    img_array = np.expand_dims(img_array, axis=0)          # Añadir dimensión batch
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)                # Normalizar
    return img_array

# 3. Función de clasificación
def classify_food(img_path):
    # Preprocesar la imagen
    processed_image = preprocess_image(img_path)
    
    # Realizar la predicción
    predictions = model.predict(processed_image)
    
    # Decodificar las predicciones
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]
    
    # Mostrar los resultados
    print("Alimentos detectados:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score:.2f})")

# 4. Ejemplo de uso
if __name__ == "__main__":
    image_path = "mantequilla.jpg"  # Ruta de la imagen
    classify_food(image_path)