import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import joblib




def town_mapping(town_map):
    town_dict = {
        'ANG MO KIO': 0, 'BEDOK': 1, 'BISHAN': 2, 'BUKIT BATOK': 3, 'BUKIT MERAH': 4,
        'BUKIT PANJANG': 5, 'BUKIT TIMAH': 6, 'CENTRAL AREA': 7, 'CHOA CHU KANG': 8,
        'CLEMENTI': 9, 'GEYLANG': 10, 'HOUGANG': 11, 'JURONG EAST': 12, 'JURONG WEST': 13,
        'KALLANG/WHAMPOA': 14, 'MARINE PARADE': 15, 'PASIR RIS': 16, 'PUNGGOL': 17,
        'QUEENSTOWN': 18, 'SEMBAWANG': 19, 'SENGKANG': 20, 'SERANGOON': 21, 'TAMPINES': 22,
        'TOA PAYOH': 23, 'WOODLANDS': 24, 'YISHUN': 25
    }
    return town_dict.get(town_map, -1)


def flat_type_mapping(flt_type):
    flat_type_dict = {
        '1 ROOM': 0, '2 ROOM': 1, '3 ROOM': 2, '4 ROOM': 3, '5 ROOM': 4,
        'EXECUTIVE': 5, 'MULTI-GENERATION': 6
    }
    return flat_type_dict.get(flt_type, -1)



def flat_model_mapping(flt_model):
    flat_model_dict = {
        'Improved': 0, 'New Generation': 1, 'Model A': 2, 'Standard': 3, 'Simplified': 4,
        'Premium Apartment': 5, 'Maisonette': 6, 'Apartment': 7, 'Model A2': 8,
        'Type S1': 9, 'Type S2': 10, 'Adjoined flat': 11, 'Terrace': 12, 'DBSS': 13,
        'Model A-Maisonette': 14, 'Premium Maisonette': 15, 'Multi Generation': 16,
        'Premium Apartment Loft': 17, 'Improved-Maisonette': 18, '2-room': 19, '3Gen': 20
    }
    return flat_model_dict.get(flt_model, -1)



def predict_price(year,town,flat_type,floor_area_sqm,flat_model,storey_start,storey_end,remaining_years,
              remaining_months,lease_commence_year):
            try:
                Year= int(year)
                Town= town_mapping(town)
                Flat_type= flat_type_mapping(flat_type)
                Floor_area_sqm= float(floor_area_sqm)
                Flat_model= flat_model_mapping(flat_model)
                Storey_start = float(storey_start)
                Storey_end = float(storey_end)
        
                if Storey_start <= 0 or Storey_end <= 0:
                     raise ValueError("Storey_start and Storey_end must be positive and greater than zero.")
                

                Storey_start= np.log(float(storey_start))
                Storey_end= np.log(float(storey_end))
                Remaining_years= int(remaining_years)
                Remaining_months= int(remaining_months)
                Lease_commence_year= int(lease_commence_year)
                
                CurrentYear =datetime.now().year
                Flat_age = CurrentYear - Lease_commence_year

            except ValueError as e:
                 st.error(f"Input error: {e}")
                 return None


            with open("Resale_Flat_Prices_Model_optimized.pkl","rb") as f1:
                regg_model= joblib.load(f1)


            Data = np.array([[Year,Town,Flat_type,Floor_area_sqm,
                                Flat_model,Storey_start,Storey_end,Remaining_years,Remaining_months,
                                Lease_commence_year, Flat_age]])
            
            st.write(f"Transformed data: {Data}")
    
            try:
                Pred = regg_model.predict(Data)
                price = np.exp(Pred[0])
                return round(price)
            except ValueError as e:
                st.error(f"Prediction error: {e}")
                return None


st.set_page_config(layout="wide")

st.title("SINGAPORE RESALE FLAT PRICES PREDICTING")
st.write("")

with st.sidebar:
    select= option_menu("MAIN MENU",["Home", "Price Prediction", "About"])

if select == "Home":

    st.header("HDB Flats:")

elif select == "Price Prediction":

    col1,col2= st.columns(2)
    with col1:

        year= st.selectbox("Select the Year",["1990","1991","1992","1993","1994","1995","1996","1997","1998",
                                              "1999","2000","2001","2002","2003","2004","2005",
                                              "2006","2007","2008","2009","2010","2011","2012","2013","2014",
                                              "2015", "2016", "2017", "2018", "2019", "2020", "2021",
                           "2022", "2023", "2024"])
        
        town= st.selectbox("Select the Town", ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                                            'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
                                            'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                                            'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
                                            'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
                                            'TOA PAYOH', 'WOODLANDS', 'YISHUN'])
        
        flat_type= st.selectbox("Select the Flat Type", ['3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', '1 ROOM',
                                                        'MULTI-GENERATION'])
        
        

        floor_area_sqm= st.number_input("Enter the Value of Floor Area sqm (Min: 31 / Max: 280")

        flat_model= st.selectbox("Select the Flat Model", ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
                                                        'Premium Apartment', 'Maisonette', 'Apartment', 'Model A2',
                                                        'Type S1', 'Type S2', 'Adjoined flat', 'Terrace', 'DBSS',
                                                        'Model A-Maisonette', 'Premium Maisonette', 'Multi Generation',
                                                        'Premium Apartment Loft', 'Improved-Maisonette', '2-room', '3Gen'])

    with col2:

        storey_start= st.number_input("Enter the Value of Storey Start(Min: 1)", min_value=1)

        storey_end= st.number_input("Enter the Value of Storey End (Min: 1)", min_value=1)

        remaining_years= st.number_input("Enter the Value of Remaining Lease Year (Min: 42 / Max: 97)")

        remaining_months= st.number_input("Enter the Value of Remaining Lease Month (Min: 0 / Max: 11)")
        
        lease_commence_year= st.selectbox("Select the Lease_Commence_Date", [str(i) for i in range(1966,2023)])

    button= st.button("Predict the Price", use_container_width= True)

    if button:
        st.write(f"Inputs: Year={year}, Town={town}, Flat_type={flat_type}, Floor Area={floor_area_sqm}, "
                f"Flat Model={flat_model}, Storey Start={storey_start}, Storey End={storey_end}, "
                f"Remaining Years={remaining_years}, Remaining Months={remaining_months}, "
                f"Lease Commence Year={lease_commence_year}")
    
    pre_price = predict_price(year, town,flat_type, floor_area_sqm, flat_model, storey_start, storey_end,
                                remaining_years, remaining_months, lease_commence_year)

    if pre_price is not None:
        st.write("## :green[**The Predicted Price is :**]", pre_price)

elif select == "About":

    st.header(":blue[Data Collection and Preprocessing:]")
    st.write("Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB) for the years 1990 to Till Date. Preprocess the data to clean and structure it for machine learning.")
