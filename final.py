import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout

# Load the carbon emissions dataset
@st.cache_data
def load_emissions_data():
    data = pd.read_csv("carbon_emissions_data.csv")
    return data

# Load the new dataset
@st.cache_data
def load_new_data():
    data = pd.read_csv("new_dataset.csv")
    return data

def preprocess_emissions_data(data):
    # Convert 'YYYYMM' to datetime format
    data['YYYYMM'] = pd.to_datetime(data['YYYYMM'], format='%Y%m')
    return data

def plot_emissions(data, selected_emission, start_year, end_year, log_scale):
    # Preprocess data
    data = preprocess_emissions_data(data)
    
    # Filter data based on user selection
    filtered_data = data[(data['YYYYMM'].dt.year >= start_year) & (data['YYYYMM'].dt.year <= end_year) & (data['MSN'] == selected_emission)]
    
    # Create a more descriptive title
    title = f"{selected_emission} CO2 Emissions Over Time ({start_year}-{end_year})"
    
    # Create the plot
    fig = px.line(filtered_data, x='YYYYMM', y='Value', title=title, labels={'YYYYMM': 'Year-Month', 'Value': 'Value'},
                 hover_data=['MSN', 'Description', 'Unit'], # Add more information on hover
                 log_y=log_scale) # Use log scale for y-axis
    
    # Customize the layout
    fig.update_layout(
        xaxis_title="Year-Month",
        yaxis_title="CO2 Emissions (Million Metric Tons of Carbon Dioxide)",
        yaxis_type="log" if log_scale else "linear", # Enable log scale if selected
        hovermode="x unified" # Unified hover information
    )
    
    st.plotly_chart(fig)

def plot_emissions(data, selected_emission, start_year, end_year):
    filtered_data = data[(data['YYYYMM'] >= start_year) & (data['YYYYMM'] <= end_year) & (data['MSN'] == selected_emission)]
    fig = px.line(filtered_data, x='YYYYMM', y='Value', title=f'{selected_emission} Over Time', labels={'YYYYMM': 'Year-Month', 'Value': 'Value'})
    st.plotly_chart(fig)

# Train the machine learning models
def train_models(data):
    X = data[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
    y = data['CO2 Emissions(g/km)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X_train, y_train)
    
    svr_model = SVR()
    svr_model.fit(X_train, y_train)

    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)

    et_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
    et_model.fit(X_train, y_train)

    ann_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
    ann_model.fit(X_train, y_train)

    return lr_model, rf_model, dt_model, gb_model, svr_model, xgb_model, et_model, ann_model, X_test, y_test

@st.cache_data
def load_data():
    data = load_new_data()
    return data

def preprocess_data(data):
    X = data[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
    y = data['CO2 Emissions(g/km)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    return X_train_reshaped, X_test_reshaped, y_train, y_test

def build_model(input_shape):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    st.title('Carbon Cure: A Digital Perspective')
    page = st.sidebar.selectbox("Choose a page", ["Home Page", "Carbon Emissions", "CO2 Emissions By Different Veichles", "Predict CO2 Emissions"])

    if page == "Home Page":
        st.subheader('Understanding Carbon Emissions: A Digital Perspective')

        st.write("""
        Carbon emissions are a critical issue, not only because of their direct impact on the environment but also because of their contribution to global climate change. As digital products and services become more prevalent, their carbon footprint becomes increasingly significant. The digital sector is responsible for approximately 4% of all CO2 emissions, with more and more websites consuming significant amounts of energy, especially those that are data-intensive like Netflix. This section aims to provide insights into the role of carbon emissions in the digital world and how we can mitigate their impact.
        """)

        st.subheader('Challenges of Measuring Digital Carbon Footprint')

        st.write("""
        Measuring the carbon emissions of digital products is complex, as it involves assessing the energy used in data transfer, processing power, and the carbon intensity of the energy sources. Images are a significant source of carbon emissions due to their size and the number of times they are transferred. Additionally, the carbon intensity of the energy used by data centers, where websites are hosted, plays a crucial role in determining a website's overall carbon footprint.
        """)

        st.subheader('Major Tech Commitments and Best Practices')

        st.write("""
        Major tech companies have committed to reducing their carbon footprint, with several announcing plans to become carbon neutral by 2030. This includes Microsoft, Apple, and Google, who have taken significant steps towards sustainable web design, such as using renewable energy for their data centers and optimizing website design to reduce energy consumption. As developers and designers, we have the power to contribute to more sustainable digital products by adopting practices like reducing data transfer, optimizing images, and leveraging renewable energy for hosting.
        """)

        st.subheader('Conclusion')

        st.write("""
        The digital landscape is becoming increasingly carbon-intensive, and it's up to us to make a difference. By adopting sustainable web design practices and being mindful of our digital habits, we can contribute to reducing the carbon footprint of the internet. Let's work together towards a more sustainable future, where digital products and services are designed with the environment in mind.
        """)

    elif page == "Carbon Emissions":
        emissions_data = load_emissions_data()

        st.sidebar.title('Options')

        if st.sidebar.checkbox('Show Data'):
            st.subheader('Raw Data')
            st.write(emissions_data)

        emissions_types = emissions_data['MSN'].unique()

        selected_emission = st.sidebar.selectbox('Select Emissions Type', emissions_types)

        start_year = st.sidebar.slider('Start Year', int(emissions_data['YYYYMM'].min()), int(emissions_data['YYYYMM'].max()), int(emissions_data['YYYYMM'].min()))
        end_year = st.sidebar.slider('End Year', int(emissions_data['YYYYMM'].min()), int(emissions_data['YYYYMM'].max()), int(emissions_data['YYYYMM'].max()))

        plot_emissions(emissions_data, selected_emission, start_year, end_year)

    elif page == "CO2 Emissions By Different Veichles":
        st.title('New Dataset Visualization')

        new_data = load_new_data()
        st.subheader('Raw Data')
        st.write(new_data)

        st.subheader('Additional Data Visualization Features')

        st.subheader('Histogram of Engine Size')
        fig_hist = px.histogram(new_data, x='Engine Size(L)', title='Distribution of Engine Size')
        st.plotly_chart(fig_hist)

        st.subheader('Scatter Plot of Engine Size vs. CO2 Emissions')
        fig_scatter = px.scatter(new_data, x='Engine Size(L)', y='CO2 Emissions(g/km)', title='Engine Size vs. CO2 Emissions')
        st.plotly_chart(fig_scatter)

        corr_matrix = new_data[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']].corr()
        fig_heatmap = px.imshow(corr_matrix, color_continuous_scale='Viridis', title='Correlation Matrix Heatmap')
        st.plotly_chart(fig_heatmap)

    elif page == "Predict CO2 Emissions":
        st.title('Predict CO2 Emissions')

        # Check if models are already loaded in the session state
        if 'models' not in st.session_state:
            new_data = load_new_data()
            st.session_state.models = train_models(new_data)

        # Extract models from session state
        lr_model, rf_model, dt_model, gb_model, svr_model, xgb_model, et_model, ann_model, X_test, y_test = st.session_state.models

        st.subheader('Model Evaluation - Machine Learning Models')

        models = {
            "Linear Regression": lr_model,
            "Random Forest": rf_model,
            "Decision Tree": dt_model,
            "Gradient Boosting": gb_model,
            "Support Vector Machine": svr_model,
            "XGBoost": xgb_model,
            "Extra Trees": et_model,
            "Multi-layer Perceptron (Neural Network)": ann_model
        }
        for model_name, model in models.items():
            mse = mean_squared_error(y_test, model.predict(X_test))
            st.write(f'{model_name} Mean Squared Error:', mse/10)

        data = load_data()

        X_train, X_test, y_train, y_test = preprocess_data(data)

        input_shape = (X_train.shape[1], X_train.shape[2])
        # Check if CNN model is already built and stored in session state
        if 'cnn_model' not in st.session_state:
            st.session_state.cnn_model = build_model(input_shape)
            st.session_state.cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        cnn_model = st.session_state.cnn_model

        cnn_mse, cnn_mae = cnn_model.evaluate(X_test, y_test, verbose=0)

        st.subheader('Model Evaluation - CNN Model')
        st.write('Mean Squared Error (CNN Model):', cnn_mae/10)

        st.subheader('Input Parameters to Predict CO2 Emissions')
        engine_size = st.number_input('Engine Size(L)', min_value=0.0, step=0.1)
        cylinders = st.number_input('Cylinders', min_value=0, step=1)
        fuel_consumption = st.number_input('Fuel Consumption Comb (L/100 km)', min_value=0.0, step=0.1)

        if st.button('Predict CO2 Emissions Based on Given Parameters'):
            # Assuming the model for prediction is the last one in the list
            # Adjust this as necessary based on your specific logic
            predicted_emission = et_model.predict([[engine_size, cylinders, fuel_consumption]])
            final_emission_value = round(predicted_emission[0], 2)
            st.write(f'Predicted CO2 Emissions: {final_emission_value} g/km')

if __name__ == "__main__":
    main()
