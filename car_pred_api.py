from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
from io import BytesIO
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
import pickle
import re

MODEL_PATH = './the_best.pkl'
ENCODER1_PATH = './encoder1.pkl'
ENCODER2_PATH = './encoder2.pkl'
SCALER1_PATH = './scaler1.pkl'
SCALER2_PATH = './scaler2.pkl'
POLY_PATH = './poly.pkl'
PATHS = [ENCODER1_PATH, ENCODER2_PATH, SCALER1_PATH, SCALER2_PATH, POLY_PATH, MODEL_PATH]

app = FastAPI()


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def clean_col(s):
    try:
        float_s = float(s.split()[0])
        return float_s
    except:
        return np.nan


def find_digit(s):
    try:
        return float(re.search('\d+', s).group())
    except:
        return np.nan


def find_max_torque(s):
    try:
        return max(map(float, re.findall('\d+', re.sub('[\.,]', '', s))))
    except:
        return np.nan


def kg_to_nm(s):
    return round(s * 9.80665, 3)


def split_torque(df, col_name='torque'):
    torque_nm = df[df[col_name].str.lower().str.contains('nm', na=False)][col_name]
    torque_kg = df[df[col_name].str.lower().str.contains('kg', na=False)][col_name]
    torque_nm_kg = df.loc[list(set(torque_nm.index).intersection(set(torque_kg.index))), col_name]
    index_nounits = df.index.difference(torque_nm.index).difference(torque_kg.index)
    torque_no_units = df.loc[index_nounits, col_name]
    torque_nm = torque_nm.loc[torque_nm.index.difference(torque_nm_kg.index)]
    torque_kg = torque_kg.loc[torque_kg.index.difference(torque_nm_kg.index)]

    torque_nm_split = torque_nm.str.split(' ', n=1, expand=True)
    if not torque_nm_split.empty:
        torque_nm_new = torque_nm_split[0].apply(find_digit).astype(float)
        torque_nm_max_new = torque_nm_split[1].apply(find_max_torque).astype(float)
    else:
        torque_nm_new = pd.Series(dtype=float)
        torque_nm_max_new = pd.Series(dtype=float)

    torque_kg_split = torque_kg.str.split(' ', n=1, expand=True)
    if not torque_kg_split.empty:
        torque_kg_new = torque_kg_split[0].apply(find_digit).apply(kg_to_nm).astype(float)
        torque_kg_max_new = torque_kg_split[1].apply(find_max_torque).astype(float)
    else:
        torque_kg_new = pd.Series(dtype=float)
        torque_kg_max_new = pd.Series(dtype=float)

    torque_nm_kg_split = torque_nm_kg.str.split(' ', n=1, expand=True)
    if not torque_nm_kg_split.empty:
        torque_nm_kg_new = torque_nm_kg_split[0].apply(find_digit).astype(float)
        torque_nm_kg_max_new = torque_nm_kg_split[1].apply(find_max_torque).astype(float)
    else:
        torque_nm_kg_new = pd.Series(dtype=float)
        torque_nm_kg_max_new = pd.Series(dtype=float)

    torque_no_units_split = torque_no_units.str.split(' ', n=1, expand=True)
    if not torque_no_units_split.empty:
        torque_no_units_new = torque_no_units_split[0].apply(find_digit).astype(float)
        torque_no_units_max_new = torque_no_units_split[1].apply(find_max_torque).astype(float)
    else:
        torque_no_units_new = pd.Series(dtype=float)
        torque_no_units_max_new = pd.Series(dtype=float)

    torque_new = pd.DataFrame(pd.concat([torque_nm_new, torque_kg_new, torque_no_units_new, torque_nm_kg_new], axis=0))

    torque_max_new = pd.DataFrame(
        pd.concat([torque_nm_max_new, torque_kg_max_new, torque_no_units_max_new, torque_nm_kg_max_new], axis=0))
    result_df = pd.concat([torque_new, torque_max_new], axis=1)
    result_df.columns = ['torque', 'max_torque_rpm']

    return result_df.sort_index()


def make_prediction(df, enc1_path, enc2_path, scaler1_path, scaler2_path, poly_path, model_path):
    df = pd.DataFrame(df)
    try:
        df = pd.DataFrame(np.repeat(df.set_index(0).T.values, 1, axis = 0), columns = df[0])
    except:
        pass
    df['year'] = df['year'].astype(int)
    df['km_driven'] = df['year'].astype(int)
    df['seats'] = df['seats'].astype(float)
    df['mileage'] = df['mileage'].apply(clean_col)
    df['max_power'] = df['max_power'].apply(clean_col)
    df['engine'] = df['engine'].apply(clean_col)

    df_torq = split_torque(df)
    df = df.drop('torque', axis=1)
    df = pd.concat([df, df_torq], axis=1)

    df_cat = df.drop(['name'], axis=1)
    ehc1 = load_model(enc1_path)
    categorical_features = df_cat.select_dtypes(include=['object']).columns
    df_con = pd.concat([df[categorical_features], df['seats'].astype(str)], axis=1)
    df_ohe = ehc1.transform(df_con)
    df_ohe_df = pd.DataFrame(df_ohe.toarray(), columns=['fuel_CNG', 'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol',
       'seller_type_Dealer', 'seller_type_Individual',
       'seller_type_Trustmark Dealer', 'transmission_Automatic',
       'transmission_Manual', 'owner_First Owner',
       'owner_Fourth & Above Owner', 'owner_Second Owner',
       'owner_Test Drive Car', 'owner_Third Owner', 'seats_10',
       'seats_14', 'seats_2', 'seats_4', 'seats_5', 'seats_6', 'seats_7',
       'seats_8', 'seats_9'])
    df_cat = df_cat.drop(categorical_features, axis=1)
    df_cat = pd.concat([df_cat, df_ohe_df.astype(int)], axis=1).drop(['seats'], axis=1)

    name = df['name']
    model_cars = name.str.split(' ', n=1, expand=True)
    df['model'] = model_cars[0]
    ehc2 = load_model(enc2_path)
    df_ohe = ehc2.transform(df['model'].to_numpy().reshape(-1, 1))
    df_models_df = pd.DataFrame(df_ohe.toarray(), columns=['model_Ambassador', 'model_Audi', 'model_BMW', 'model_Chevrolet',
       'model_Daewoo', 'model_Datsun', 'model_Fiat', 'model_Force',
       'model_Ford', 'model_Honda', 'model_Hyundai', 'model_Isuzu',
       'model_Jaguar', 'model_Jeep', 'model_Kia', 'model_Land',
       'model_Lexus', 'model_MG', 'model_Mahindra', 'model_Maruti',
       'model_Mercedes-Benz', 'model_Mitsubishi', 'model_Nissan',
       'model_Peugeot', 'model_Renault', 'model_Skoda', 'model_Tata',
       'model_Toyota', 'model_Volkswagen', 'model_Volvo']).astype(int)
    df_cat_models = pd.concat([df_cat, df_models_df], axis=1)

    scaler2 = load_model(scaler2_path)

    df_only_num = df_cat_models[['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']]
    df_cat_models = df_cat_models.drop(
        ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm'], axis=1)
    df_scaled = scaler2.transform(df_only_num)
    df_scaled = pd.DataFrame(df_scaled, columns=df_only_num.columns)
    df_scaled = pd.concat([df_scaled, df_cat_models], axis=1)

    poly = load_model(poly_path)

    df_only_num = df_scaled[['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']]
    df_cat_models = df_scaled.drop(['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm'],
                                   axis=1)
    df_scaled = poly.transform(df_only_num)
    columns_poly = poly.get_feature_names_out(df_only_num.columns)
    df_scaled = pd.DataFrame(df_scaled, columns=columns_poly)
    df_scaled = pd.concat([df_scaled, df_cat_models], axis=1)

    elastic_model = load_model(model_path)

    return elastic_model.predict(df_scaled)


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    pred = make_prediction(item, *PATHS)

    return pred


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)):
    df = pd.read_csv(BytesIO(file.file.read()))
    predictions = make_prediction(df, *PATHS)
    df['predictions'] = predictions
    output_file = BytesIO()
    df.to_csv(output_file, index=False)
    output_file.seek(0)
    response = StreamingResponse(output_file, media_type="text/csv",
                                headers={'Content-Disposition': 'attachment; filename=predictions.csv'})
    return response