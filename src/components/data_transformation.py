import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.components.data_ingestion import DataIngestion
from src.constants import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, TARGET_COLUMN
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    @staticmethod
    def get_data_transformer_object():
        """
        This method is used for performing data transformation on the dataset
        """
        try:

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])

            cat_pipeline = Pipeline(

                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ])

            logging.info("Numerical columns are: {}".format(NUMERICAL_COLUMNS))
            logging.info("Categorical columns are: {}".format(CATEGORICAL_COLUMNS))

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, NUMERICAL_COLUMNS),
                    ('cat_pipeline', cat_pipeline, CATEGORICAL_COLUMNS)
                ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path: str, test_data_path: str):

        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Read train and test dataframes")

            logging.info("Obtaining the preprocessor object")

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_test_df = test_df[TARGET_COLUMN]

            logging.info("Applying the preprocessor object on the train and test dataframes")

            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_array, np.array(target_train_df)]
            test_arr = np.c_[input_feature_test_array, np.array(target_test_df)]

            logging.info("Saved preprocessor object")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path
                    )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data_path=train_data, test_data_path=test_data)
