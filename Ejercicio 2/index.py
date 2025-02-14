import tensorflow as tf
import numpy as np

# Base de conocimiento inicial
base_conocimiento = {
    "objetos": {
        "manzana": {"color": "rojo", "forma": "esferica", "textura": "lisa"},
        "zanahoria": {"color": "naranja", "forma": "alargada", "textura": "rugosa"},
        "platano": {"color": "amarillo", "forma": "curva", "textura": "lisa"}
    },
    "reglas": [
        {
            "condiciones": ["color == 'rojo'", "forma == 'esferica'", "textura == 'lisa'"],
            "conclusion": "manzana",
            "probabilidad": 0.9
        },
        {
            "condiciones": ["color == 'naranja'", "forma == 'alargada'", "textura == 'rugosa'"],
            "conclusion": "zanahoria",
            "probabilidad": 0.8
        },
        {
            "condiciones": ["color == 'amarillo'", "forma == 'curva'", "textura == 'lisa'"],
            "conclusion": "platano",
            "probabilidad": 0.85
        }
    ]
}

# Función para evaluar condiciones
def evaluar_condiciones(condiciones, atributos):
    return all(eval(condicion, {}, atributos) for condicion in condiciones)

# Motor de inferencia
def motor_inferencia(atributos_objeto):
    resultados = []
    for regla in base_conocimiento["reglas"]:
        if evaluar_condiciones(regla["condiciones"], atributos_objeto):
            resultados.append({
                "objeto": regla["conclusion"],
                "probabilidad": regla["probabilidad"]
            })
    return resultados

# Manejo de incertidumbre con TensorFlow
def calcular_probabilidad_final(resultados):
    probabilidades = [resultado["probabilidad"] for resultado in resultados]
    tensor_probabilidades = tf.constant(probabilidades, dtype=tf.float32)
    probabilidad_final = tf.reduce_max(tensor_probabilidades).numpy()
    return probabilidad_final

# Interfaz de usuario
def interfaz_usuario():
    print("Bienvenido al Sistema Experto Clasificador de Objetos")
    atributos_objeto = {}
    atributos_objeto["color"] = input("¿De qué color es el objeto? (rojo/naranja/amarillo): ").lower()
    atributos_objeto["forma"] = input("¿Qué forma tiene el objeto? (esferica/alargada/curva): ").lower()
    atributos_objeto["textura"] = input("¿Qué textura tiene el objeto? (lisa/rugosa): ").lower()
    
    resultados = motor_inferencia(atributos_objeto)
    if resultados:
        probabilidad_final = calcular_probabilidad_final(resultados)
        print(f"El objeto es probablemente un(a) {resultados[0]['objeto']} con probabilidad {probabilidad_final:.2f}")
    else:
        print("No se pudo determinar la clasificación del objeto con los datos proporcionados.")

# Ejecución del sistema
if __name__ == "__main__":
    interfaz_usuario()