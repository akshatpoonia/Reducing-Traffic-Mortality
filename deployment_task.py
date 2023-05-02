import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import streamlit as st
from sklearn.pipeline import Pipeline


#Importing the dataset
df = pd.read_csv('modeling.csv')

#take a random sample of 40000 rows
df = df.sample(n=40000, random_state=42)

# Function to add Year_driven and average_mileage columns
def add_year_and_mileage(df):
    current_year = datetime.now().year
    df['Years_Ago'] = current_year - df['Year']  
    df['Avg_Mileage_Per_Year'] = df['Mileage'] / df['Years_Ago']
    return df

#function for frequency encoding
def frequency_encoding(df, columns):
    for col in columns:
        freq = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq)
    return df

cat_columns = ['State', 'City', 'Model']

# # Split the data into training and testing sets
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Apply frequency encoding to the data
df_encoded = frequency_encoding(df, cat_columns)

# print(df_encoded)

# # Apply frequency encoding to training data
X_train_encoded = frequency_encoding(X_train, cat_columns)
# print(X_train_encoded)

# MinMax scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
# print(X_train_scaled)
# print(y_train)


# # PCA for dimensionality reduction
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)
# print(X_train_pca)
# print(y_train)
# Train a Random Forest Regression model on training data
rf = RandomForestRegressor()
rf.fit(X_train_pca, y_train)

# # Apply frequency encoding and scaling to testing data
X_test_encoded = frequency_encoding(X_test, cat_columns)
X_test_scaled = scaler.transform(X_test_encoded)
X_test_pca = pca.transform(X_test_scaled)
# print(X_test_pca)
# Predict the price for the testing data using the trained model
y_pred = rf.predict(X_test_pca)
# print(y_pred)


#save the rf model into a pickle file
import pickle
pickle.dump(rf, open('rf.pkl', 'wb'))

#streamlight
us_states = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
    "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
    "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas",
    "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
]


us_cities = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego",
    "Dallas", "San Jose", "Austin", "Jacksonville", "Fort Worth", "Columbus", "San Francisco", "Charlotte",
    "Indianapolis", "Seattle", "Denver", "Washington", "Boston", "Nashville", "El Paso", "Detroit", "Memphis",
    "Portland", "Oklahoma City", "Las Vegas", "Louisville", "Baltimore", "Milwaukee", "Albuquerque", "Tucson",
    "Fresno", "Sacramento", "Mesa", "Atlanta", "Kansas City", "Colorado Springs", "Miami", "Raleigh", "Omaha",
    "Long Beach", "Virginia Beach", "Oakland", "Minneapolis", "Tulsa", "Wichita", "New Orleans", "Arlington"
]

car_models = [
    'Acura ILX', 'Audi A3', 'BMW 2 Series', 'Buick Encore', 'Cadillac ATS',
    'Chevrolet Camaro', 'Chrysler 300', 'Dodge Challenger', 'Ford Mustang',
    'Genesis G70', 'GMC Acadia', 'Honda Accord', 'Hyundai Sonata',
    'Infiniti Q50', 'Jeep Wrangler', 'Kia Sorento', 'Lexus ES', 'Lincoln Aviator',
    'Mazda CX-5', 'Mercedes-Benz C-Class', 'Nissan Altima', 'Porsche 911',
    'Ram 1500', 'Subaru Outback', 'Toyota Camry', 'Volkswagen Arteon',
    'Volvo S60', 'Tesla Model 3', 'Lucid Air', 'Rivian R1T', 'Chevrolet Silverado',
    'Ford F-150', 'GMC Sierra', 'Ram 2500', 'Toyota Tundra', 'Nissan Titan',
    'Jeep Gladiator', 'Tesla Cybertruck', 'Rivian R1S', 'Ford Ranger', 'GMC Canyon'
]

# col1, col2, col3 = st.columns(3)

# with col1:
State = st.selectbox('Select the State',sorted(us_states))
# with col2:
City = st.selectbox('Select the Citites',sorted(us_cities))
# with col3:
Model = st.selectbox('Select the Car Model',sorted(car_models))

Mileage = st.number_input('Mileage')
Year = st.number_input('Year')

#Create a dataframe taking the values
new_df = pd.DataFrame({'Year':[Year],
                        'Mileage':[Mileage],
                        'City':[City],
                        'State':[State],
                        'Model':[Model]})
# st.write(new_df)

# Encode the categorical features using the same frequency encoding
new_df_encoded = frequency_encoding(new_df, cat_columns)
new_df_encoded = add_year_and_mileage(new_df_encoded)
# Scale the new data using the same scaler
new_df_scaled = scaler.transform(new_df_encoded)

# Apply PCA to the new data
new_df_pca = pca.transform(new_df_scaled)

# Predict the price for the new car using the trained model
new_price = rf.predict(new_df_pca)
# print(new_price)

# Add a predict button
if st.sidebar.button("Predict"):
    # Call the predict_new_price function to get the new price
    new_price = rf.predict(new_df_pca)

    # Display the new price in bold
    st.write(f"Price of the car is **{new_price}**")




