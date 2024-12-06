import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization

class TitanicModelTrainer:
    def __init__(self, data_path):
        # Load the CSV data into a pandas DataFrame
        self.data = pd.read_csv(data_path)
        
        # Split features (X) and target (y)
        self.X = self.data.drop('Survived', axis=1)  # All columns except 'Survived'
        self.y = self.data['Survived']               # Only the 'Survived' column
        
        # Add preprocessing
        self.X = self.X.fillna(self.X.mean())  # Simple mean imputation
        
        # Create train/test splits
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,                # Features
            self.y,                # Target
            test_size=0.2,         # 20% for testing, 80% for training
            stratify=self.y,       # Maintain same ratio of survived/died in both splits
            random_state=42        # For reproducibility
        )

    def _compute_class_weights(self):
        """Compute class weights for imbalanced dataset"""
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(self.y_train)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=self.y_train
        )
        return dict(zip(classes, weights))

    def train_logistic_regression(self):
        # Initialize model with L2 regularization and balanced class weights
        model = LogisticRegression(
            max_iter=1000,          # Maximum iterations for convergence
            class_weight='balanced', # Handles imbalanced classes
            random_state=42         # For reproducibility
        )
        
        # Get learning curves
        train_sizes, train_scores, test_scores = learning_curve(
            model,                                    # The model to evaluate (LogisticRegression or DecisionTree)
            self.X_train, self.y_train,              # Training data and labels
            train_sizes=np.linspace(0.1, 1.0, 10),   # Creates 10 evenly spaced points between 10% and 100% of training data
            cv=5                                      # 5-fold cross-validation
        )
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.title('Learning Curves (Logistic Regression)')
        plt.legend(loc='best')
        plt.grid(True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'logistic_regression_learning_curves_{timestamp}.png')
        plt.close()
        
        # Train final model and record loss history
        model.fit(self.X_train, self.y_train)  # Fit the model first
        
        # Record loss history for a fixed number of iterations (e.g., 100)
        n_iterations = 100
        losses = []
        for i in range(n_iterations):
            model_i = LogisticRegression(max_iter=i+1, class_weight='balanced', random_state=42)
            model_i.fit(self.X_train, self.y_train)
            train_loss = -model_i.score(self.X_train, self.y_train)
            test_loss = -model_i.score(self.X_test, self.y_test)
            losses.append((train_loss, test_loss))
        
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        train_losses, test_losses = zip(*losses)
        plt.plot(range(1, len(losses) + 1), train_losses, label='Training Loss')
        plt.plot(range(1, len(losses) + 1), test_losses, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Curves (Logistic Regression)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(f'logistic_regression_loss_curves_{timestamp}.png')
        plt.close()
        
        # Evaluate
        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)
        y_pred = model.predict(self.X_test)
        
        # Print results
        print("\nLogistic Regression Results:")
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = f"""Logistic Regression Results:
Training accuracy: {train_score:.4f}
Test accuracy: {test_score:.4f}

Classification Report:
{classification_report(self.y_test, y_pred)}"""
        
        with open(f'logistic_regression_report_{timestamp}.txt', 'w') as f:
            f.write(report)
        
        # Save model with timestamp and accuracy
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_info = {
            'model': model,
            'timestamp': timestamp,
            'test_accuracy': test_score,
            'train_accuracy': train_score,
            'feature_names': self.X.columns.tolist(),
            'X_train_stats': {
                'mean': self.X_train.mean(),
                'std': self.X_train.std()
            }
        }
        joblib.dump(model_info, f'logistic_regression_{timestamp}_{test_score:.4f}.joblib')
        
        return model

    def train_decision_tree(self):
        # Initialize model with early stopping criteria
        model = DecisionTreeClassifier(
            max_depth=5,  # Prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        
        # Get learning curves
        train_sizes, train_scores, test_scores = learning_curve(
            model, self.X_train, self.y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5
        )
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.title('Learning Curves (Decision Tree)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('decision_tree_learning_curves.png')
        plt.close()
        
        # Record loss history
        n_iterations = 20  # Fewer iterations since we're varying max_depth
        losses = []
        for i in range(1, n_iterations + 1):
            model_i = DecisionTreeClassifier(max_depth=i, class_weight='balanced', random_state=42)
            model_i.fit(self.X_train, self.y_train)
            train_loss = -model_i.score(self.X_train, self.y_train)
            test_loss = -model_i.score(self.X_test, self.y_test)
            losses.append((train_loss, test_loss))
        
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        train_losses, test_losses = zip(*losses)
        plt.plot(range(1, len(losses) + 1), train_losses, label='Training Loss')
        plt.plot(range(1, len(losses) + 1), test_losses, label='Validation Loss')
        plt.xlabel('Max Depth')
        plt.ylabel('Loss')
        plt.title('Loss Curves (Decision Tree)')
        plt.legend(loc='best')
        plt.grid(True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'decision_tree_loss_curves_{timestamp}.png')
        plt.close()
        
        # Train final model
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)
        y_pred = model.predict(self.X_test)
        
        print("\nDecision Tree Results:")
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = f"""Decision Tree Results:
Training accuracy: {train_score:.4f}
Test accuracy: {test_score:.4f}

Classification Report:
{classification_report(self.y_test, y_pred)}"""
        
        with open(f'decision_tree_report_{timestamp}.txt', 'w') as f:
            f.write(report)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        importances = pd.Series(model.feature_importances_, index=self.X.columns)
        importances.sort_values().plot(kind='barh')
        plt.title('Feature Importance (Decision Tree)')
        plt.savefig('decision_tree_feature_importance.png')
        plt.close()
        
        # Save model with timestamp and accuracy
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_info = {
            'model': model,
            'timestamp': timestamp,
            'test_accuracy': test_score,
            'train_accuracy': train_score
        }
        joblib.dump(model_info, f'decision_tree_{timestamp}_{test_score:.4f}.joblib')
        
        return model
    def train_random_forest(self):
        # Initialize model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5, 
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        
        # Get learning curves
        train_sizes, train_scores, test_scores = learning_curve(
            model, self.X_train, self.y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5
        )
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.title('Learning Curves (Random Forest)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('random_forest_learning_curves.png')
        plt.close()
        
        # Record loss history
        n_iterations = 100  # Testing different numbers of trees
        losses = []
        for i in range(1, n_iterations + 1, 5):  # Step by 5 to reduce computation
            model_i = RandomForestClassifier(n_estimators=i, max_depth=5, class_weight='balanced', random_state=42)
            model_i.fit(self.X_train, self.y_train)
            train_loss = -model_i.score(self.X_train, self.y_train)
            test_loss = -model_i.score(self.X_test, self.y_test)
            losses.append((train_loss, test_loss))
        
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        train_losses, test_losses = zip(*losses)
        plt.plot(range(1, len(losses)*5 + 1, 5), train_losses, label='Training Loss')
        plt.plot(range(1, len(losses)*5 + 1, 5), test_losses, label='Validation Loss')
        plt.xlabel('Number of Trees')
        plt.ylabel('Loss')
        plt.title('Loss Curves (Random Forest)')
        plt.legend(loc='best')
        plt.grid(True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'random_forest_loss_curves_{timestamp}.png')
        plt.close()
        
        # Train final model
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)
        y_pred = model.predict(self.X_test)
        
        print("\nRandom Forest Results:")
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = f"""Random Forest Results:
Training accuracy: {train_score:.4f}
Test accuracy: {test_score:.4f}

Classification Report:
{classification_report(self.y_test, y_pred)}"""
        
        with open(f'random_forest_report_{timestamp}.txt', 'w') as f:
            f.write(report)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        importances = pd.Series(model.feature_importances_, index=self.X.columns)
        importances.sort_values().plot(kind='barh')
        plt.title('Feature Importance (Random Forest)')
        plt.savefig('random_forest_feature_importance.png')
        plt.close()
        
        # Save model with timestamp and accuracy
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_info = {
            'model': model,
            'timestamp': timestamp,
            'test_accuracy': test_score,
            'train_accuracy': train_score
        }
        joblib.dump(model_info, f'random_forest_{timestamp}_{test_score:.4f}.joblib')
        
        return model

    def train_xgboost(self):
        # Initialize model without early_stopping_rounds in constructor
        model = XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Get learning curves (using basic model without early stopping)
        train_sizes, train_scores, test_scores = learning_curve(
            model, self.X_train, self.y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5
        )
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.title('Learning Curves (XGBoost)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('xgboost_learning_curves.png')
        plt.close()
        
        # For the actual training, create a new model with early stopping
        final_model = XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=20
        )
        
        # Create evaluation set for early stopping
        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        
        # Train model with early stopping
        final_model.fit(
            self.X_train, 
            self.y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        # Get the best number of trees
        best_iteration = final_model.best_iteration
        
        # Plot loss curves using the evaluation history
        results = final_model.evals_result()
        epochs = len(results['validation_0']['logloss'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(epochs), results['validation_0']['logloss'], label='Training Loss')
        plt.plot(range(epochs), results['validation_1']['logloss'], label='Validation Loss')
        plt.axvline(x=best_iteration, color='r', linestyle='--', label='Best Iteration')
        plt.xlabel('Number of Boosting Rounds')
        plt.ylabel('Log Loss')
        plt.title('Learning Curves with Early Stopping (XGBoost)')
        plt.legend(loc='best')
        plt.grid(True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'xgboost_loss_curves_{timestamp}.png')
        plt.close()
        
        # Evaluate
        train_score = final_model.score(self.X_train, self.y_train)
        test_score = final_model.score(self.X_test, self.y_test)
        y_pred = final_model.predict(self.X_test)
        
        print("\nXGBoost Results:")
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        print(f"Best iteration found: {best_iteration}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = f"""XGBoost Results:
Training accuracy: {train_score:.4f}
Test accuracy: {test_score:.4f}
Best iteration: {best_iteration}

Classification Report:
{classification_report(self.y_test, y_pred)}"""
        
        with open(f'xgboost_report_{timestamp}.txt', 'w') as f:
            f.write(report)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        importances = pd.Series(final_model.feature_importances_, index=self.X.columns)
        importances.sort_values().plot(kind='barh')
        plt.title('Feature Importance (XGBoost)')
        plt.savefig('xgboost_feature_importance.png')
        plt.close()
        
        # Save model with timestamp and accuracy
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_info = {
            'model': final_model,
            'timestamp': timestamp,
            'test_accuracy': test_score,
            'train_accuracy': train_score,
            'best_iteration': best_iteration
        }
        joblib.dump(model_info, f'xgboost_{timestamp}_{test_score:.4f}.joblib')
        
        return final_model
    
    def train_deep_learning(self):
        # Initialize model with better architecture
        model = Sequential([
            Dense(256, activation='relu', input_shape=(self.X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])

        # Use learning rate scheduling
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True)
        
        # Compile with better optimizer settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # Enhanced early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            min_delta=0.001
        )
        
        # Add reduce LR on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001
        )
        
        # Train with better settings
        history = model.fit(
            self.X_train, self.y_train,
            epochs=200,  # Increased epochs since we have early stopping
            batch_size=64,  # Larger batch size
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            class_weight=self._compute_class_weights(),  # Add class weights
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'deep_learning_history_{timestamp}.png')
        plt.close()
        
        # Evaluate
        train_score = model.evaluate(self.X_train, self.y_train, verbose=0)[1]
        test_score = model.evaluate(self.X_test, self.y_test, verbose=0)[1]
        y_pred = (model.predict(self.X_test) > 0.5).astype(int)
        
        print("\nDeep Learning Results:")
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = f"""Deep Learning Results:
Training accuracy: {train_score:.4f}
Test accuracy: {test_score:.4f}

Classification Report:
{classification_report(self.y_test, y_pred)}"""
        
        with open(f'deep_learning_report_{timestamp}.txt', 'w') as f:
            f.write(report)
            
        # Save model
        model.save(f'deep_learning_model_{timestamp}_{test_score:.4f}.h5')
        
        return model


if __name__ == "__main__":
    trainer = TitanicModelTrainer('data/train_clean.csv')
    log_reg_model = trainer.train_logistic_regression()
    # decision_tree_model = trainer.train_decision_tree()
    # random_forest_model = trainer.train_random_forest()
    # xgboost_model = trainer.train_xgboost()
    # deep_learning_model = trainer.train_deep_learning()