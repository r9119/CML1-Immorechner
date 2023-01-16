import asyncio
from hypercorn.config import Config
from hypercorn.asyncio import serve

from typing import Union
from fastapi import FastAPI
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import load

## Pipeline functions
numeric_features_reduced = ['living_area', 'rooms_combined', 'Year_built']

numeric_features_reduced_cls = ['living_area', 'rooms_combined', 'Year_built', 'price']

categorical_features_reduced = ['Type', 'plz']

categorical_features_reduced_cls = ['plz']

class ImproveLivinAreaReduced(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
        
    def fit(self, X, y = None):
        return self
    
    # function for merging columns
    def ColumnMerger2(self, row):
        return row.Living_area_unified

    def transform(self, X, y = None):
        # merge columns
        X['living_area'] = X.apply(self.ColumnMerger2, axis=1)
        
        return X

class ImproveRoomsReduced(BaseEstimator, TransformerMixin):
    # initializer 
    def __init__(self):
        pass
        
    def fit(self, X, y = None):
        return self

    # function for merging the columns
    def ColumnMerger(self, row):
        return row.rooms

    def transform(self, X, y = None):

        X.rooms = X.rooms.astype(str)
        
        # clean up rooms column and convert to float
        X['rooms'] = (
            X.rooms.str.replace("rm", "")
            .str.replace(",", ".")
            .str.replace(" ", "")
            .astype(float)
        )

        # merge columns into new one
        X['rooms_combined'] = X.apply(self.ColumnMerger, axis=1)

        return X

class TypeCleaner(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):

        # define mapping for replacement
        housing_types_map = {
            'Apartment': 'flat',
            'Bifamiliar house': 'detached-house',
            'Terrace flat': 'stepped-apartment',
            'Single house': 'detached-house',
            'Multi-family house': 'detached-house',
            'Duplex': 'duplex-maisonette',
            'Row house': 'semi-detached-house',
            'Attic flat': 'attic-flat',
            'Furnished dwelling': 'furnished-residential-property',
            'Villa': 'villa',
            'Loft': 'loft',
            'Studio': 'studio',
            'Bachelor flat': 'studio',
            'Roof flat': 'penthouse',
            'Farm house': 'farmhouse',
            'Castle': 'castle',
            'Terrace house': 'terrace-house',
            'Rustic house': 'rustico',
            'Rustico':'rustico',
            'Chalet': 'chalet',
            'Attic': 'attic-room',
            'Hobby room': 'single-room',
            'Granny flat': 'flat',
            'Cellar compartment': 'single-room',
            'Single Room': 'single-room',
        }

        # replace types
        X['Type'] = X['Type'].replace(housing_types_map)

        return X

class NumericStripper(BaseEstimator, TransformerMixin):
    
    def __init__(self, numeric_features):
        
        self.numeric_features = numeric_features

        self.floor_map= {
            "m2": "",
            " ": "",
            ",": "",
            "floor": "",
            "m²": "",
            "4. Basement": "-4",
            "3. Basement": "-3",
            "2. Basement": "-2",
            "1. Basement": "-1",
            "Ground floor": "0",
            "100. floor": "nan",
            "999. floor": "nan",
            "nan": "nan",
            "Ground": "nan",
            "2.Basement": "-2",
            "4.Basement": "-4",
            "1.Basement": "-1",
            "3.Basement": "-3",
            "GF": "0",
            "m": ""
        }
        
    def fit(self, X, y = None):
        return self

    def strip_to_numeric(self, value):
        # check if the value is a string
        if isinstance(value, str):
        # replace the whitespaces with a bottomline
            for key, replacement in self.floor_map.items():
                value = value.replace(key, replacement)
            if value.__contains__('. floor'):
                value = value.replace('. floor', "")
            if value.__contains__('%'):
                value = value.replace('%', "")
                value = float(value)/100
            return float(value)

        # if the value is not a string, return it as is
        return value

    def transform(self, X, y = None):
        # strip numeric features from strings
        X[self.numeric_features] = X[self.numeric_features].applymap(self.strip_to_numeric).astype(float)
        
        return X

class FeaturePreselectorReducedRegressor(BaseEstimator, TransformerMixin):
     
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        # reduce dataframe to selected features
        X = pd.concat([X[numeric_features_reduced], X[categorical_features_reduced]], axis=1) 
        return X

class FeaturePreselectorReducedClassifier(BaseEstimator, TransformerMixin):
     
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        # reduce dataframe to selected features
        X = pd.concat([X[numeric_features_reduced_cls], X[categorical_features_reduced_cls]], axis=1)
        return X

class predictor(object):
    ''' Immobilienrechner predictor - Price and Type
    '''

    def __init__(self, type = None):
        self.type = type
        self.classifier = np.load('smallClsModel.npy', allow_pickle=True).item()
        self.regressor = np.load('smallRegModel.npy', allow_pickle=True).item()
        self.classifierPipeline = load('./preprocessing_pipeline_reduced_cls.joblib')
        self.regressionPipeline = load('./preprocessing_pipeline_reduced_reg.joblib')
        self.houseTypeArray = ['attic-flat', 'attic-room', 'chalet', 'detached-house', 'duplex-maisonette', 'farmhouse', 'flat', 'furnished-residential-property', 'loft', 'penthouse', 'rustico', 'semi-detached-house', 'stepped-apartment', 'stepped-house', 'studio', 'terrace-house', 'villa']

    def regression_pred(self, x):
        ''' Predict the price of the house
        '''
        # Structure data for prediction - onehot encoding etc
        x_pd = pd.DataFrame([x], columns=['Living_area_unified', 'rooms', 'Year_built', 'plz', 'Type'])
        x_transformed = self.regressionPipeline.transform(x_pd)

        # Predict the price
        prediction = self.regressor.predict(x_transformed)

        return np.around(prediction, -4)

    def classification_pred(self, x):
        ''' Predict the type of the house
        '''
        # Structure data for prediction - onehot encoding etc
        x_pd = pd.DataFrame([x], columns=['Living_area_unified', 'rooms', 'Year_built', 'plz', 'price'])
        x_transformed = self.regressionPipeline.transform(x_pd)

        # Predict the type
        prediction = self.classifier.predict(x_transformed)

        return self.houseTypeArray[int(prediction)]

predictor = predictor()

app = FastAPI()

@app.get("/")
def read_root():
    return {"root": "you are at root"}

@app.post("/predictor/")
async def predict(
    pred_type: str,
    area: Union[int, float] = None,
    rooms: int = None,
    year: int = None,
    plz: int = None,
    price: Union[int, float] = None,
    house_type: str = None
):
    if pred_type == "reg": # Regressior - Price
        x = [area, rooms, year, plz, house_type]

        prediction = predictor.regression_pred(x)

        return {"Predicted price": prediction}

    elif pred_type == "clr": # Classifier - House type
        x = [area, rooms, year, plz, price]

        prediction = predictor.classification_pred(x)

        return {"Predicted house type": prediction}
    
    return {"Error": "No predictor under this name"}

asyncio.run(serve(app, Config()))