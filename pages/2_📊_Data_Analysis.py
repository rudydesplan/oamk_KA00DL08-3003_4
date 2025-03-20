# pages/2_📊_Data_Analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split

@st.cache_data
def load_data():
    df = pd.read_csv('ames_housing.csv')
    df['House Age'] = df['Yr Sold'] - df['Year Built']
    df.drop(['Lot Area','Year Built', 'Year Remod', 'Garage Finish', 
            'Mo Sold', 'Yr Sold', 'Full Bath'], axis=1, inplace=True)
    return df

@st.cache_data
def preprocess_data(df):
    # Encoding and feature engineering
    qual_dict = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    functional_map = {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0}
    
    data = df.copy()
    data['Exter Qual'] = data['Exter Qual'].map(qual_dict)
    data['Kitchen Qual'] = data['Kitchen Qual'].map(qual_dict)
    data['Functional'] = data['Functional'].map(functional_map)
    
    nominal_features = ['Neighborhood', 'Sale Type', 'Sale Condition']
    data = pd.get_dummies(data, columns=nominal_features, drop_first=True, dtype=int)
    data.dropna(inplace=True)
    
    X = data.drop(columns=['SalePrice'])
    y = data['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test, data

def main():
    st.title("Exploratory Data Analysis")
    
    try:
        df = load_data()
        X_train, X_test, y_train, y_test, processed_data = preprocess_data(df)
        
        # Numerical Analysis
        st.header("Numerical Variables")
        num_cols = ['Overall Qual', 'Overall Cond', 'Mas Vnr Area', 'Total Bsmt SF',
                   '1st Flr SF', 'Gr Liv Area', 'Bsmt Full Bath', 'Bedroom AbvGr',
                   'Kitchen AbvGr', 'TotRms AbvGrd', 'Garage Cars', 'Garage Area',
                   'SalePrice', 'House Age']
        
        col1, col2 = st.columns(2)
        with col1:
            var = st.selectbox("Select Numerical Variable", num_cols)
            fig = px.box(df, x=var, title=f"Boxplot of {var}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x=var, nbins=30, title=f"Distribution of {var}")
            fig.add_vline(x=df[var].mean(), line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        # Categorical Analysis
        st.header("Categorical Variables")
        cat_cols = ['Neighborhood', 'Exter Qual', 'Kitchen Qual',
                   'Functional', 'Sale Type', 'Sale Condition']
        
        cat_var = st.selectbox("Select Categorical Variable", cat_cols)
        counts = df[cat_var].value_counts().reset_index()
        fig = px.bar(counts, x='index', y=cat_var, title=f"Distribution of {cat_var}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Bivariate Analysis
        st.header("Bivariate Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            feat = st.selectbox("Select Feature", [c for c in num_cols if c != 'SalePrice'])
            fig = px.scatter(df, x=feat, y='SalePrice', trendline="lowess")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            cat_feat = st.selectbox("Select Categorical Feature", cat_cols)
            fig = px.violin(df, x=cat_feat, y='SalePrice', box=True)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main()