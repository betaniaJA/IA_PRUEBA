import tensorflow as tf
import numpy as np
import random
from collections import deque

# ==============================
# Configuración inicial del sistema
# ==============================

class ObjectDetectionEnv:
    def __init__(self):
        self.detection_threshold = 0.7
        self.current_frame = None
        self.detections = []
        self.user_feedback = None
        self.available_operators = [
            'apply_detection_model',
            'adjust_threshold',
            'query_external_db',
            'change_roi',
            'random_action'
        ]
    
    def reset(self, frame):
        self.current_frame = frame
        self.detections = []
        return self.get_state()
    
    def get_state(self):
        return {
            'frame': self.current_frame,
            'detections': self.detections,
            'threshold': self.detection_threshold,
            'user_feedback': self.user_feedback
        }

# ==============================
# Modelos y operadores
# ==============================
class DetectionSystem:
    @staticmethod
    def mock_detection_model(frame, threshold):
        # Simulación de detección de objetos
        objects = ['person', 'car', 'dog']
        return [(obj, random.uniform(0.5, 0.95)) for obj in objects if random.random() > 0.3]
    
    @staticmethod
    def query_external_database(detections):
        # Simulación de consulta externa
        return [d for d in detections if d[1] > 0.6]

# ==============================
# Sistema de Búsqueda Inteligente
# ==============================
class IntelligentSearchAgent:
    def __init__(self, state_size=100, action_size=5):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # Factor de descuento
        self.epsilon = 0.5  # Exploración inicial
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Modelo de utilidad con TensorFlow
        self.utility_model = self.build_utility_model()
        
    def build_utility_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
        return model
    
    def state_to_features(self, state):
        # Convertir estado a características numéricas
        return np.concatenate([
            [state['threshold']],
            [len(state['detections'])],
            [state['user_feedback'] if state['user_feedback'] else 0]
        ]).reshape(1, -1)
    
    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size-1)
        
        state_features = self.state_to_features(state)
        act_values = self.utility_model.predict(state_features, verbose=0)
        return np.argmax(act_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((
            self.state_to_features(state),
            action,
            reward,
            self.state_to_features(next_state),
            done
        ))
    
    def retrain_model(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.utility_model.predict(next_state, verbose=0)[0])
            
            target_f = self.utility_model.predict(state, verbose=0)
            target_f[0][action] = target
            
            self.utility_model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==============================
# Simulación del proceso completo
# ==============================
def simulate_full_process():
    # Inicialización de componentes
    env = ObjectDetectionEnv()
    agent = IntelligentSearchAgent()
    
    # Simular entrada de video
    frames = [np.random.rand(224, 224, 3) for _ in range(10)]  # 10 frames de ejemplo
    
    for frame in frames:
        state = env.reset(frame)
        total_reward = 0
        done = False
        
        while not done:
            # Selección de acción
            action_idx = agent.select_action(state)
            action = env.available_operators[action_idx]
            
            # Aplicar operador
            if action == 'apply_detection_model':
                new_detections = DetectionSystem.mock_detection_model(
                    state['frame'], 
                    state['threshold']
                )
                env.detections = new_detections
                reward = len(new_detections) * 0.1
                
            elif action == 'adjust_threshold':
                env.detection_threshold = max(0.3, min(0.9, state['threshold'] + random.uniform(-0.1, 0.1)))
                reward = -0.05
                
            elif action == 'query_external_db':
                validated = DetectionSystem.query_external_database(state['detections'])
                env.detections = validated
                reward = len(validated) * 0.15
                
            elif action == 'change_roi':
                # Simular cambio de región de interés
                reward = 0.1
                
            elif action == 'random_action':
                # Acción aleatoria para interacción
                env.detections = random.sample(state['detections'], 
                                             min(3, len(state['detections'])))
                reward = 0.2
            
            # Verificar condición de terminación
            done = (len(env.detections) >= 3) or (random.random() < 0.1)  # Condiciones simuladas
            
            # Obtener feedback de usuario simulado
            env.user_feedback = random.choice([-1, 0, 1])  # -1: negativo, 0: neutral, 1: positivo
            reward += env.user_feedback * 0.3
            
            # Almacenar experiencia y reentrenar
            next_state = env.get_state()
            agent.remember(state, action_idx, reward, next_state, done)
            agent.retrain_model()
            
            total_reward += reward
            state = next_state
        
        print(f"Frame procesado. Recompensa total: {total_reward:.2f}")
        print(f"Detecciones finales: {state['detections']}")
        print("=====================================")

# ==============================
# Interfaz de usuario simulada
# ==============================
if __name__ == "__main__":
    # Inicializar componentes de TensorFlow
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # Ejecutar simulación completa
    simulate_full_process()