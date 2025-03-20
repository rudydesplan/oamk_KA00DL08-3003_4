# pages/p5_Neural_Network.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from pages.p2_Data_Analysis import load_data, preprocess_data

@st.cache_resource
def train_nn_model(_X_train, _y_train):
    # StandardScaler integration
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(_X_train)
    
    # Functional API implementation
    input_dim = X_train_scaled.shape[1]
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Custom optimizer configuration
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(delta=calculate_optimal_delta(_y_train)),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    
    # Early stopping integration
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    model.fit(
        X_train_scaled, _y_train,
        validation_split=0.2,
        epochs=250,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model, scaler

def calculate_optimal_delta(y):
    q25, q75 = np.percentile(y, [25, 75])
    iqr = q75 - q25
    return 1.5 * iqr  # Tukey's fence multiplier

def main():
    st.title("Neural Network Analysis")
    
    try:
        df = load_data()
        X_train, X_test, y_train, y_test, processed_data = preprocess_data(df)
        
        # Train model with scaler
        model, scaler = train_nn_model(X_train, y_train)
        
        # Scale test data
        X_test_scaled = scaler.transform(X_test)
        
        # Metrics
        y_pred = model.predict(X_test_scaled).flatten()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RMSE", f"${rmse:,.2f}")
        with col2:
            st.metric("R-squared", f"{r2:.4f}")
        
        # Permutation importance
        st.subheader("Feature Importance (Permutation)")
        
        # Create wrapper for model with scaler
        class ScaledModel:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
            
            def predict(self, X):
                return self.model.predict(self.scaler.transform(X))

            def fit(self, X, y=None):
                return self
        
        with st.spinner("Calculating permutation importance..."):
            results = permutation_importance(
                ScaledModel(model, scaler),
                X_test,
                y_test,
                n_repeats=5,
                random_state=42,
                scoring='neg_root_mean_squared_error'
            )
            
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': results.importances_mean * -1
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title='Neural Network Feature Importance (Permutation)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Interpretation:
        - Measures how much RMSE increases when a feature is randomized
        - Higher values mean the feature is more critical to predictions
        - Directly comparable to other models' importance metrics
        """)
        st.dataframe(importance_df)
        
    except Exception as e:
        st.error(f"Error in neural network analysis: {str(e)}")

if __name__ == "__main__":
    main()
