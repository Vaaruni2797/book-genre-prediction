import os
import sys
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class LabelEncodingConfig:
    label_encoder_obj_file_path = os.path.join('artifacts', "label_encoder.pkl")


class LabelEncoding:
    def __init__(self):
        self.label_encoding_config = LabelEncodingConfig()

    def get_label_encoding_object(self):
        '''
        This function is responsible for encoding the given genres
        '''
        try:
            label_encoder = LabelEncoder()
            logging.info("Applying Label Encoding to the target variable")
            return label_encoder
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_label_encoding(self, y_train, y_test):
        '''
        This function applies the label encoding object to the given genre
        '''
        try:
            label_encoding_obj = self.get_label_encoding_object()
            y_train_encoded = label_encoding_obj.fit_transform(y_train)
            y_test_encoded = label_encoding_obj.transform(y_test)

            # Save the preprocessing object only if training
            save_object(
                file_path=self.label_encoding_config.label_encoder_obj_file_path,
                obj=label_encoding_obj
            )

            return y_train_encoded, y_test_encoded

        except Exception as e:
            raise CustomException(e, sys)