# pages/3_📈_Linear_Regression.py
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pages.2_📊_Data_Analysis import load_data, preprocess_data

@st.cache_resource
def train_linear_model(_X_train, _y_train):
    model = LinearRegression()
    model.fit(_X_train, _y_train)
    return model

def main():
    st.title("Linear Regression Analysis")
    
    try:
        df = load_data()
        X_train, X_test, y_train, y_test, processed_data = preprocess_data(df)
        model = train_linear_model(X_train, y_train)
        
        # Metrics
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RMSE", f"${rmse:,.2f}")
        with col2:
            st.metric("R-squared", f"{r2:.4f}")
        
        # OLS Summary
        st.subheader("Statistical Summary")
        X_train_sm = sm.add_constant(X_train)
        sm_model = sm.OLS(y_train, X_train_sm).fit()
        st.text(str(sm_model.summary()))
        
        # Feature importance
        st.subheader("Top 15 Predictive Features")
        coef_df = pd.DataFrame({
            'Feature': sm_model.params.index[1:],
            'Coefficient': sm_model.params.values[1:],
            'P-value': sm_model.pvalues.values[1:]
        })
        
        meaningful_features = coef_df[(coef_df['P-value'] < 0.05)].copy()
        meaningful_features['Absolute_Impact'] = meaningful_features['Coefficient'].abs()
        top15 = meaningful_features.sort_values('Absolute_Impact', ascending=False).head(15)
        
        fig = px.bar(top15, x='Coefficient', y='Feature', orientation='h',
                    title='Top 15 Most Meaningful Features Impacting SalePrice')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top15.style.format({'Coefficient': '${:,.0f}', 'P-value': '{:.4f}'}))
        
    except Exception as e:
        st.error(f"Error in linear regression analysis: {str(e)}")

if __name__ == "__main__":
    main()