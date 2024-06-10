import tensorflow as tensorflow

def load_model(model_path: str):
    return tensorflow.keras.models.load_model(model_path)