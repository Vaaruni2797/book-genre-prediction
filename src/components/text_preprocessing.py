import os
import sys
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class TextPreprocessingConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "text_preprocessor.pkl")


class TextPreprocessing:
    def __init__(self):
        self.text_preprocessing_config = TextPreprocessingConfig()

    def get_text_preprocessor_object(self):
        '''
        This function is responsible for creating the text preprocessing pipeline
        '''
        try:
            text_pipeline = Pipeline(
                steps=[
                    ('clean_text', CleanTextTransformer()),
                    ('remove_stopwords', RemoveStopwordsTransformer()),
                    ('lemmatize', LemmatizeTextTransformer()),
                    ('stem', StemTextTransformer())
                ]
            )
            logging.info("Text preprocessing pipeline creation completed")
            return text_pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_text_preprocessing(self, text_data):
        '''
        This function applies the preprocessing object to the given text data
        '''
        try:
            logging.info("Starting text preprocessing")
            
            text_preprocessing_obj = self.get_text_preprocessor_object()

            text_data['summary'] = text_preprocessing_obj.fit_transform(text_data['summary'])
            
            logging.info("Text preprocessing completed")

            save_object(
                file_path=self.text_preprocessing_config.preprocessor_obj_file_path,
                obj=text_preprocessing_obj
            )
            
            return text_data, self.text_preprocessing_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)


# Custom Transformers
class CleanTextTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.clean(text) for text in X]

    def clean(self, text):
        text = re.sub("\'", "", text)
        text = re.sub("[^a-zA-Z]", " ", text)
        text = ' '.join(text.split())
        text = text.lower()
        return text


class RemoveStopwordsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        stop_words = set(stopwords.words('english'))
        return [' '.join([word for word in text.split() if word not in stop_words]) for text in X]


class LemmatizeTextTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        lemma = WordNetLemmatizer()
        return [' '.join([lemma.lemmatize(word) for word in text.split()]) for text in X]


class StemTextTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        stemmer = PorterStemmer()
        return [' '.join([stemmer.stem(word) for word in text.split()]) for text in X]