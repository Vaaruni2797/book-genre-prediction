import os
import sys
from dataclasses import dataclass
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score 

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array['summary'],
                train_array['genre'],
                test_array['summary'],
                test_array['genre']
            )

            # Step 1: Label Encoding the target variable (genre)
            logging.info("Applying Label Encoding to the target variable")
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)

            # Step 2: TF-IDF Vectorization
            logging.info("Applying TF-IDF vectorization")
            tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
            
            # Fit and transform on training data
            X_train_tfidf = tfidf_vectorizer.fit_transform(X_train.values.astype('U'))
            # Transform on test data
            X_test_tfidf = tfidf_vectorizer.transform(X_test.values.astype('U'))

            # Step 3: Model training and evaluation
            models = {
                "Logistic Regressor": LogisticRegression(),
                "Support Vector Machine (linear kernel)": svm.SVC(kernel='linear', probability=True),
                "Support Vector Classifier (rbf kernel)": svm.SVC(kernel='rbf', gamma=1, probability=True)
            }

            model_report: dict = evaluate_models(
                X_train=X_train_tfidf,
                y_train=y_train_encoded,  # Use encoded labels
                X_test=X_test_tfidf,
                y_test=y_test_encoded,  # Use encoded labels
                models=models
            )
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found is {best_model_name}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Step 4: Model evaluation
            predicted = best_model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test_encoded, predicted)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)