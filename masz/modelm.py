from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression


class ModelManager:


    def __init__(self):
        self.model = None

    def train_random_forest(self, X_train, y_train):
        """7. Model Training"""
        print("\n[INFO] Training Random Forest...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def train_with_optimization(self, X_train, y_train):
        """8. Fine Tuning (Hyperparameter Optimization)"""
        print("\n[INFO] Fine tuning (GridSearch)...")
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
        }
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        print(f"Best Parameters: {grid_search.best_params_}")
        self.model = grid_search.best_estimator_

    def evaluate(self, X_test, y_test):
        """10. Analysis and Evaluation"""
        print("\n--- 10. Evaluation Results ---")
        y_pred = self.model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    def compare_models(self, X_train, y_train, X_test, y_test):
        """8. Alternative Classifier Comparison"""
        print("\n--- Comparison with Logistic Regression ---")
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        print(f"Logistic Regression Accuracy: {lr.score(X_test, y_test):.4f}")
        print(f"Random Forest Accuracy:       {self.model.score(X_test, y_test):.4f}")