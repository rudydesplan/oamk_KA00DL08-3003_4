# pages/p5_Neural_Network.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from pages.p2_Data_Analysis import load_data, preprocess_data

class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    """Wrapper class to make Keras models compatible with scikit-learn"""
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        
    def fit(self, X, y):
        """Dummy fit method to satisfy sklearn interface"""
        return self  # Model is already trained
    
    def predict(self, X):
        """Make predictions using the scaled input"""
        return self.model.predict(self.scaler.transform(X)).flatten()

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
    
    return KerasRegressorWrapper(model, scaler)  # Return wrapped model

def calculate_optimal_delta(y):
    q25, q75 = np.percentile(y, [25, 75])
    iqr = q75 - q25
    return 1.5 * iqr  # Tukey's fence multiplier

def main():
    st.title("Neural Network Analysis")
    
    try:
        df = load_data()
        X_train, X_test, y_train, y_test, processed_data = preprocess_data(df)
        
        # Train model
        model = train_nn_model(X_train, y_train)
        
        # Metrics
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RMSE", f"${rmse:,.2f}")
        with col2:
            st.metric("R-squared", f"{r2:.4f}")
        
        # Permutation importance
        st.subheader("Feature Importance (Permutation)")
        with st.spinner("Calculating permutation importance..."):
            results = permutation_importance(
                model,
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
        - Shows average RMSE increase when features are randomized
        - Higher values indicate more important features
        - Directly comparable to other models' feature importance
        """)
        st.dataframe(importance_df)
        
    except Exception as e:
        st.error(f"Error in neural network analysis: {str(e)}")

if __name__ == "__main__":
    main()
