import pandas as pd
import pickle
import numpy as np
import json


class PredictCancellation(object):
    def __init__(self):
        self.average_ticket = pickle.load(open(     '../../parameter/average_ticket_scaler.pkl', 'rb'))
        self.avg_price_per_room = pickle.load(open( '../../parameter/avg_price_per_room_scaler.pkl', 'rb'))
        self.lead_time = pickle.load(open(          '../../parameter/lead_time_scaler.pkl', 'rb'))
        self.market_segment_type = pickle.load(open('../../parameter/market_segment_type_encoder.pkl', 'rb'))
        self.price_category = pickle.load(open(     '../../parameter/price_category_encoder.pkl', 'rb'))
        self.room_type_reserved = pickle.load(open( '../../parameter/room_type_reserved_encoder.pkl', 'rb'))
        self.type_of_meal_plan = pickle.load(open(  '../../parameter/type_of_meal_plan_encoder.pkl', 'rb'))

    def convert_binary_columns(self, df):
        binary_cols = ['repeated_guest', 'required_car_parking_space']
        df[binary_cols] = df[binary_cols].astype(bool)
        return df
    
    def feature_engineering(self, df):
        df = df.copy()

        # Criar colunas derivadas
        df['no_guests'] = df['no_of_children'] + df['no_of_adults']
        df['no_days'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
        df['average_ticket'] = df['avg_price_per_room'] / df['no_guests']
        df['is_family'] = ((df['no_of_children'] > 0) & (df['no_of_adults'] >= 2)).astype(bool)

        # Ajustar booking_month
        df['booking_month'] = df['arrival_month'] - (df['lead_time'] // 30)
        df['booking_month'] = df['booking_month'].apply(lambda x: x + 12 if x <= 0 else x)

        # Prevenir erro de qcut com uma linha só ou valores idênticos
        if df['avg_price_per_room'].nunique() < 4:
            df['price_category'] = 'mid-low'  # valor neutro default
        else:
            df['price_category'] = pd.qcut(
                df['avg_price_per_room'],
                q=4,
                labels=['low', 'mid-low', 'mid-high', 'high'],
                duplicates='drop'  # se houver valores repetidos
            )

        return df

    
    def data_preparation(self, df):
        df = df.copy()

        def safe_encode(column, encoder):
            return column.apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

        df['room_type_reserved'] = safe_encode(df['room_type_reserved'], self.room_type_reserved)
        df['type_of_meal_plan'] = safe_encode(df['type_of_meal_plan'], self.type_of_meal_plan)
        df['market_segment_type'] = safe_encode(df['market_segment_type'], self.market_segment_type)
        df['price_category'] = safe_encode(df['price_category'], self.price_category)

        # Escalonamento numérico
        df['lead_time'] = self.lead_time.transform(df[['lead_time']])
        df['avg_price_per_room'] = self.avg_price_per_room.transform(df[['avg_price_per_room']])
        df['average_ticket'] = self.average_ticket.transform(df[['average_ticket']])

        boruta_columns = [
            'lead_time',
            'arrival_month',
            'arrival_date',
            'market_segment_type',
            'avg_price_per_room',
            'no_of_special_requests',
            'average_ticket',
            'booking_month'
        ]

        return df[boruta_columns]

    
    def get_predictions(self, model, test_data,original_data):
        pred= model.predict(test_data)
        original_data['prediction'] = pred
        
        return original_data.to_json(orient='records')
