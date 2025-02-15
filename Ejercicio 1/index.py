import tensorflow as tf
import numpy as np

# Cargar modelo preentrenado
model =  tf.keras.applications.MobileNetV2(weights='imagenet')

def cargar_y_preprocesar_imagen(ruta_imagen):
    img = tf.keras.preprocessing.image.load_img(ruta_imagen, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def clasificar_ingrediente(imagen):
    preds = model.predict(imagen)
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=6)[0]
    return [(label, score) for (_, label, score) in decoded_preds]

def interactuar_con_usuario(predicciones):
    print("Predicciones:")
    for i, (label, score) in enumerate(predicciones):
        print(f"{i + 1}. {label} ({score:.2f})")
    respuesta = input("¿Alguna de estas predicciones es correcta? (s/n): ")
    return respuesta.lower() == 's'

def main():
    ruta_imagen = input("Ingrese la ruta de la imagen: ")
    imagen = cargar_y_preprocesar_imagen(ruta_imagen)
    
    max_iteraciones = 5
    iteracion = 0
    
    while iteracion < max_iteraciones:
        predicciones = clasificar_ingrediente(imagen)
        if interactuar_con_usuario(predicciones):
            print("¡Ingrediente identificado correctamente!")
            break
        else:
            print("Refinando resultados...")
        iteracion += 1
    
    if iteracion == max_iteraciones:
        print("Límite de iteraciones alcanzado. Intente nuevamente con otra imagen.")

if __name__ == "__main__":
    main()