import os
import sys
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class VectorizingConfig:
    vectorizer_obj_file_path = os.path.join('artifacts', "vectorizer.pkl")


class Vectorizing:
    def __init__(self):
        self.vectorizing_config = VectorizingConfig()

    def get_vectorizing_object(self):
        '''
        This function is responsible for vectorizing the given text
        '''
        try:
            vector = TfidfVectorizer(max_df=0.8, max_features=10000)
            logging.info("Vectorizing object creation completed")
            return vector
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_vectorizing(self, train_array, test_array):
        '''
        This function applies the vectorizing object to the given text data.
        '''
        try:
            vectorizing_obj = self.get_vectorizing_object()
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array['summary'],
                train_array['genre'],
                test_array['summary'],
                test_array['genre']
            )

            logging.info("Applying TF-IDF vectorization")
            
            # Fit and transform on training data
            X_train_tfidf = vectorizing_obj.fit_transform(X_train.values.astype('U'))
            # Transform on test data
            X_test_tfidf = vectorizing_obj.transform(X_test.values.astype('U'))

            # Save the preprocessing object only if training
            save_object(
                file_path=self.vectorizing_config.vectorizer_obj_file_path,
                obj=vectorizing_obj
            )

            return X_train_tfidf, y_train, X_test_tfidf,y_test

        except Exception as e:
            raise CustomException(e, sys)