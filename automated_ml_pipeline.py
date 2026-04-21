
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import shap

# 1. CUSTOM TRANSFORMER: CORRELATION PRUNING
class CorrelationDropper(BaseEstimator, TransformerMixin):
    """
    Reduces the hypothesis space by removing multicollinear features.
    Ensures model identifiability by eliminating features with high correlation.
    """
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop = None

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        corr_matrix = X.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Find features with correlation greater than threshold
        self.to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return X.drop(columns=self.to_drop)

# 2. DATA PREPARATION
def load_and_prepare_data():
    # Simulating the dataset (50K samples, 25 features)
    X_raw, y = make_classification(
        n_samples=50000, 
        n_features=25, 
        n_informative=15, 
        n_redundant=5, 
        random_state=42
    )
    feature_names = [f'feat_{i}' for i in range(25)]
    X = pd.DataFrame(X_raw, columns=feature_names)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 3. END-TO-END PIPELINE CONSTRUCTION: X' = S(T(X))
def build_pipeline(X):
    numeric_features = X.columns.tolist()
    
    # Transformation T (Imputation) and S (Scaling)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features)]
    )

    # Full Pipeline Architecture
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('variance_selector', VarianceThreshold(threshold=0.01)), # Var(Xj) > epsilon
        ('corr_dropper', CorrelationDropper(threshold=0.85)),     # Correlation Pruning
        ('mi_selector', SelectKBest(score_func=mutual_info_classif)), # MI Maximization
        ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
    ])

# 4. TRAINING AND ERM OPTIMIZATION
def train_model(pipeline, X_train, y_train):
    param_grid = {
        'mi_selector__k': [10, 15, 20],
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring='accuracy', 
        n_jobs=-1, 
        verbose=1
    )
    
    print("Starting Empirical Risk Minimization (Grid Search)...")
    grid_search.fit(X_train, y_train)
    return grid_search

# 5. EXECUTION
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    pipeline = build_pipeline(X_train)
    
    results = train_model(pipeline, X_train, y_train)
    best_model = results.best_estimator_
    
    print(f"\nBest Parameters: {results.best_params_}")
    print(f"Best CV Accuracy: {results.best_score_:.4f}")
    
    # Evaluation
    y_pred = best_model.predict(X_test)
    print("\nFinal Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Model Persistence
    joblib.dump(best_model, 'automated_ml_pipeline.pkl')
    print("\nModel serialized successfully.")
