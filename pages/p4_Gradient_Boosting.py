# pages/p4_Gradient_Boosting.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pages.p2_Data_Analysis import load_data, preprocess_data

@st.cache_resource
def train_gb_model(_X_train, _y_train):
    model = GradientBoostingRegressor(n_estimators=1024, learning_rate=0.1, random_state=42)
    model.fit(_X_train, _y_train)
    return model

def main():
    st.title("Gradient Boosting Analysis")
    
    try:
        df = load_data()
        X_train, X_test, y_train, y_test, processed_data = preprocess_data(df)
        model = train_gb_model(X_train, y_train)
        
        # Metrics
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RMSE", f"${rmse:,.2f}")
        with col2:
            st.metric("R-squared", f"{r2:.4f}")
        
        # Feature importance
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title='Feature Importances from Gradient Boosting')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Interpretation:
        - Measures relative contribution of each feature to predictions
        - Values sum to 1.0 (100% of predictive power)
        """)
        st.dataframe(importance_df.style.format({'Importance': '{:.3f}'}))
        
    except Exception as e:
        st.error(f"Error in gradient boosting analysis: {str(e)}")

if __name__ == "__main__":
    main()