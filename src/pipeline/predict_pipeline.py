import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.normpath(os.path.join("artifacts","text_preprocessor.pkl"))
            vectorizer_path=os.path.normpath(os.path.join("artifacts","vectorizer.pkl"))
            label_encoder_path=os.path.normpath(os.path.join("artifacts","label_encoder.pkl"))

            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            vectorizer = load_object(file_path=vectorizer_path)
            label_encoder = load_object(file_path=label_encoder_path)

            data_scaled = preprocessor.transform(features)
            data_vectorized = vectorizer.transform(data_scaled)
            
            pred_proba = model.predict_proba(data_vectorized)
            top_pred_idx = pred_proba.argmax(axis=1)
            predicted_genre = label_encoder.inverse_transform(top_pred_idx)
            
            all_genres = label_encoder.classes_
            sorted_probabilities = sorted(zip(pred_proba[0], all_genres), reverse=True)

            return predicted_genre, sorted_probabilities
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, 
                 title:str,
                 summary:str):
        self.title = title
        self.summary = summary

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "title":[self.title],
                "summary": [self.summary]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)