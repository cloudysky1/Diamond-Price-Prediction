import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        try:
            # relative paths to artifacts
            self.preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            self.model_path = os.path.join('artifacts','model.pkl')

            # load objects
            self.preprocessor = load_object(self.preprocessor_path)
            self.model = load_object(self.model_path)

        except Exception as e:
            logging.info("Exception occurred while loading model/preprocessor")
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            # transform features
            data_scaled = self.preprocessor.transform(features)

            # predict
            pred = self.model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe gathered')
            return df
        except Exception as e:
            logging.info('Exception occurred in CustomData')
            raise CustomException(e, sys)
