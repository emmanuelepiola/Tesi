import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks, CondensedNearestNeighbour
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

class SensorRandomForestModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.ensemble_model = None
        self.base_models = {}
        self.ensemble_mode = False
        self.minority_classes = []
        self.majority_classes = []
        self.scaler = StandardScaler()
        self.label_mapping = {
            0: 'ACETIC_ACID', 1: 'ACETONE', 2: 'AIR', 3: 'AMMONIA', 4: 'AMMONIUM_CHLORIDE',
            5: 'APPLE_VINEGAR', 6: 'BALSAMIC_VINEGAR', 7: 'BIOETHANOL', 8: 'BUTANE', 
            9: 'CALCIUM_NITRATE', 10: 'DIESEL', 11: 'ETHANOL', 12: 'FORMIC_ACID', 
            13: 'GASOLINE', 14: 'HYDROGEN_PEROXIDE', 15: 'ISOPROPANOL', 16: 'KEROSENE', 
            17: 'LIGHTER_FLUID', 18: 'METHANE', 19: 'NITROMETHANE', 20: 'PHOSPHORIC_ACID', 
            21: 'RED_WINE', 22: 'SODIUM_HYDROXIDE', 23: 'UREA', 24: 'WATER_VAPOR'
        }
        
    def load_data(self, file_path="data/preprocessed_data.pkl"):
        """Load the preprocessed sensor data"""
        try:
            print(f"Loading data from {file_path}...")
            data = pd.read_pickle(file_path)
            print(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"❌ File {file_path} not found!")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def analyze_class_distribution(self, y):
        """Analyze class distribution and identify problematic classes"""
        print("📊 Class Distribution Analysis:")
        class_counts = pd.Series(y).value_counts().sort_index()
        
        for label_num, count in class_counts.items():
            chemical_name = self.label_mapping[label_num]
            percentage = (count / len(y)) * 100
            print(f"  {chemical_name}: {count} samples ({percentage:.1f}%)")
        
        # Identify minority classes (less than 2% of data)
        minority_threshold = len(y) * 0.02
        minority_classes = []
        for label_num, count in class_counts.items():
            if count < minority_threshold:
                minority_classes.append(self.label_mapping[label_num])
        
        if minority_classes:
            print(f"⚠️ Minority classes detected: {minority_classes}")
            print("💡 Consider using class_weight='balanced' for better performance")
        
        return class_counts

    def prepare_data(self, df, test_size=0.2, scale_features=True, feature_engineering=True, balance_classes=True, balance_method='smote', manual_feature_selection=True, downscale_majority=True, target_classes=None):
        """
        Prepara i dati per il training dividendo in train/test set con preprocessing avanzato
        
        Args:
            df: DataFrame con i dati
            test_size: Percentuale del test set (default 0.2)
            scale_features: Se applicare StandardScaler (default True)
            feature_engineering: Se applicare feature engineering avanzate (default True)
            balance_classes: Se bilanciare le classi (default True) 
            balance_method: Metodo di bilanciamento ('smote', 'adasyn', 'borderline', 'smote_tomek', 'smote_enn')
            manual_feature_selection: Se applicare la selezione manuale delle feature (default True)
            downscale_majority: Se ridimensionare le classi maggioritarie (default True)
            target_classes: Lista specifica di classi da bilanciare (es. [7, 8, 18] per BIOETHANOL, BUTANE, METHANE)
        """
        
        # Analyze class distribution
        self.analyze_class_distribution(df['LABEL'])
        
        # Feature engineering
        if feature_engineering:
            print("🔧 Applying feature engineering...")
            df_enhanced = self.apply_feature_engineering(df.copy())
        else:
            df_enhanced = df.copy()
        
        # Separate features and target
        X = df_enhanced.drop(columns=['LABEL'])
        y = df_enhanced['LABEL']
        
        print(f"📈 Features after engineering: {X.shape[1]} features")
        
        # Manual feature selection
        if manual_feature_selection:
            print("🎯 Applying manual feature selection...")
            X_selected = self.select_specific_features(X)
        else:
            X_selected = X
        
        # Store feature names for later use in plotting
        self.feature_names = list(X_selected.columns)
        
        print(f"📊 Final feature count: {X_selected.shape[1]} features")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Convert to numpy arrays to ensure consistency and avoid feature name warnings
        X_train_values = X_train.values if hasattr(X_train, 'values') else X_train
        X_test_values = X_test.values if hasattr(X_test, 'values') else X_test
        
        # Balance classes if requested
        if balance_classes:
            X_train_balanced, y_train_balanced = self.balance_classes_comprehensive(
                X_train_values, y_train, balance_method=balance_method, downscale_majority=downscale_majority, target_classes=target_classes
            )
        else:
            X_train_balanced = X_train_values
            y_train_balanced = y_train
        
        # Scale features if requested
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train_balanced)
            X_test_scaled = self.scaler.transform(X_test_values)
            return X_train_scaled, X_test_scaled, y_train_balanced, y_test
        else:
            return X_train_balanced, X_test_values, y_train_balanced, y_test
    
    def apply_feature_engineering(self, df):
        """Apply feature engineering to improve model performance"""
        # Get sensor columns (exclude LABEL)
        sensor_cols = [col for col in df.columns if col != 'LABEL']
        
        # 1. Add ratio features between different sensor types
        print("  Adding ratio features...")
        for i, col1 in enumerate(sensor_cols[:12]):  # First 12 are SnAu sensors
            for col2 in sensor_cols[12:16]:  # Last 4 are AlOx sensors
                if df[col2].abs().mean() > 0.1:  # Avoid division by very small numbers
                    ratio_name = f"RATIO_{col1.split('_')[2]}_{col2.split('_')[2]}"
                    df[ratio_name] = df[col1] / (df[col2] + 1e-6)  # Add small epsilon
        
        # 2. Add polynomial features for important sensors
        print("  Adding polynomial features...")
        important_sensors = sensor_cols[:6]  # First 6 sensors seem most important
        for col in important_sensors:
            df[f"{col}_SQUARED"] = df[col] ** 2
            df[f"{col}_SQRT"] = np.sqrt(np.abs(df[col]))
        
        # 3. Add statistical features across sensor groups
        print("  Adding statistical features...")
        snau_200_cols = [col for col in sensor_cols if '200.0' in col]
        snau_78125_cols = [col for col in sensor_cols if '78125.0' in col]
        alox_cols = [col for col in sensor_cols if 'ALUMINUM_OXIDE' in col]
        
        # Group statistics
        df['SNAU_200_MEAN'] = df[snau_200_cols].mean(axis=1)
        df['SNAU_200_STD'] = df[snau_200_cols].std(axis=1)
        df['SNAU_78125_MEAN'] = df[snau_78125_cols].mean(axis=1)
        df['SNAU_78125_STD'] = df[snau_78125_cols].std(axis=1)
        df['ALOX_MEAN'] = df[alox_cols].mean(axis=1)
        df['ALOX_STD'] = df[alox_cols].std(axis=1)
        
        # 4. Add interaction features
        print("  Adding interaction features...")
        df['SNAU_INTERACTION'] = df['SNAU_200_MEAN'] * df['SNAU_78125_MEAN']
        df['TEMP_INTERACTION'] = df[snau_200_cols[0]] * df[snau_200_cols[-1]]  # First and last temperature
        
        # Replace any NaN or inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())
        
        print(f"  ✅ Feature engineering complete: {df.shape[1] - 1} features")
        return df
    
    def select_specific_features(self, X):
        """
        Select only the specific most important features identified from analysis
        """
        # Define the specific features to keep
        target_features = [
            'ONCHIP_ALUMINUM_OXIDE_OUT7_IN-PHASE',
            'ALOX_MEAN',
            'OFFCHIP_SENSIMOX_SnAu_150_78125.0_IN-PHASE',
            'ONCHIP_ALUMINUM_OXIDE_OUT7_QUADRATURE',
            'ALOX_STD',
            'RATIO_SnAu_OXIDE',
            'OFFCHIP_SENSIMOX_SnAu_400_78125.0_IN-PHASE',
            'OFFCHIP_SENSIMOX_SnAu_150_200.0_IN-PHASE',
            'ONCHIP_ALUMINUM_OXIDE_OUT4_QUADRATURE',
            'ONCHIP_ALUMINUM_OXIDE_OUT4_IN-PHASE',
            'SNAU_200_STD',
            'OFFCHIP_SENSIMOX_SnAu_150_200.0_IN-PHASE_SQUARED',
            'OFFCHIP_SENSIMOX_SnAu_150_200.0_IN-PHASE_SQRT',
            'OFFCHIP_SENSIMOX_SnAu_300_78125.0_IN-PHASE',
            'OFFCHIP_SENSIMOX_SnAu_350_78125.0_IN-PHASE',
            'OFFCHIP_SENSIMOX_SnAu_200_78125.0_IN-PHASE'
        ]
        # provvisorio
        target_features = [
            "ALOX_MEAN",
            "ONCHIP_ALUMINUM_OXIDE_OUT7_IN-PHASE",
            "OFFCHIP_SENSIMOX_SnAu_150_78125.0_IN-PHASE",
            "ONCHIP_ALUMINUM_OXIDE_OUT7_QUADRATURE",
            "ALOX_STD",
            "RATIO_SnAu_OXIDE",
            "OFFCHIP_SENSIMOX_SnAu_150_200.0_IN-PHASE",
            "OFFCHIP_SENSIMOX_SnAu_400_78125.0_IN-PHASE",
            "ONCHIP_ALUMINUM_OXIDE_OUT4_QUADRATURE",
            "ONCHIP_ALUMINUM_OXIDE_OUT4_IN-PHASE",
            "OFFCHIP_SENSIMOX_SnAu_150_200.0_IN-PHASE_SQUARED",
            "OFFCHIP_SENSIMOX_SnAu_150_200.0_IN-PHASE_SQRT",
            "SNAU_200_STD",
            "OFFCHIP_SENSIMOX_SnAu_300_78125.0_IN-PHASE",
            "OFFCHIP_SENSIMOX_SnAu_400_200.0_IN-PHASE",
            "OFFCHIP_SENSIMOX_SnAu_350_78125.0_IN-PHASE"
        ]

        
        available_features = list(X.columns)
        selected_features = []
        
        print(f"  Available features: {len(available_features)}")
        print(f"  Target features: {len(target_features)}")
        
        # Find matching features (exact match or close match)
        for target_feature in target_features:
            if target_feature in available_features:
                selected_features.append(target_feature)
                print(f"    ✅ Found: {target_feature}")
            else:
                # Try to find close matches
                close_matches = [f for f in available_features if self._feature_similarity(target_feature, f)]
                if close_matches:
                    # Use the best match
                    best_match = close_matches[0]
                    selected_features.append(best_match)
                    print(f"    🔧 Close match: {target_feature} → {best_match}")
                else:
                    print(f"    ❌ Not found: {target_feature}")
        
        if not selected_features:
            print("  ⚠️ No target features found! Using all available features.")
            return X
        
        # Create DataFrame with selected features
        X_selected = X[selected_features].copy()
        
        print(f"  ✅ Selected {len(selected_features)} out of {len(target_features)} target features")
        print(f"  📊 Feature reduction: {len(available_features)} → {len(selected_features)} features")
        
        return X_selected
    
    def _feature_similarity(self, target, candidate):
        """
        Check if two feature names are similar (to handle minor naming variations)
        """
        # Normalize feature names for comparison
        target_norm = target.upper().replace('SNAU', 'SNAU').replace('_', '').replace('-', '')
        candidate_norm = candidate.upper().replace('SNAU', 'SNAU').replace('_', '').replace('-', '')
        
        # Exact match after normalization
        if target_norm == candidate_norm:
            return True
        
        # Check if candidate contains most of the target (for partial matches)
        if len(target_norm) > 10:  # Only for longer feature names
            similarity_threshold = 0.8
            matching_chars = sum(1 for a, b in zip(target_norm, candidate_norm) if a == b)
            similarity = matching_chars / max(len(target_norm), len(candidate_norm))
            return similarity >= similarity_threshold
        
        return False
    
    def train_model(self, X_train, y_train):
        """Train Random Forest model with optimized parameters"""
        print("🌳 Training Random Forest with optimized parameters...")
        
        # Create model with optimized parameters for high accuracy
        self.model = RandomForestClassifier(
            n_estimators=250,  # More trees for better performance
            max_depth=18,      # Deeper trees
            min_samples_split=2,  # Allow more splits
            min_samples_leaf=1,   # More granular leaves
            max_features=0.8,     # Use more features
            class_weight='balanced',  # Handle class imbalance
            bootstrap=True,
            oob_score=True,      # Out-of-bag scoring
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )

        # Train the model
        start_time = time.time()
        n_trees = self.model.n_estimators

        with tqdm(total=n_trees, desc="Training Trees", unit="tree") as pbar:
            self.model.fit(X_train, y_train)
            pbar.n = pbar.total
            pbar.refresh()

        end_time = time.time()
        print(f"✅ Training completed in {end_time - start_time:.2f} seconds")

        # Print OOB score if available
        if hasattr(self.model, 'oob_score_') and self.model.oob_score_ is not None:
            print(f"📊 Out-of-bag score: {self.model.oob_score_:.4f}")
        
        print("🎯 Model training completed!")
    
    def train_ensemble_model(self, X_train, y_train, minority_threshold=0.05):
        """Train powerful stacking ensemble model with multiple state-of-the-art algorithms"""
        print("🚀 Training Advanced Stacking Ensemble Model...")
        print("🎯 Algorithms: XGBoost, LightGBM, CatBoost, RandomForest, ExtraTrees + Meta-Learner")
        
        # Analyze class distribution to identify minority classes
        class_counts = pd.Series(y_train).value_counts()
        total_samples = len(y_train)
        
        self.minority_classes = []
        self.majority_classes = []
        
        for class_id, count in class_counts.items():
            percentage = count / total_samples
            if percentage < minority_threshold:
                self.minority_classes.append(class_id)
            else:
                self.majority_classes.append(class_id)
        
        print(f"🎯 Minority classes ({len(self.minority_classes)}): {[self.label_mapping[c] for c in self.minority_classes]}")
        print(f"🏢 Majority classes ({len(self.majority_classes)}): {[self.label_mapping[c] for c in self.majority_classes]}")
        
        # Define base models with optimized hyperparameters
        print("\n🧠 Setting up base models...")
        
        # XGBoost - Extremely powerful gradient boosting
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=1,  # Will be handled by class_weight
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
            eval_metric='mlogloss'
        )
        
        # LightGBM - Fast and accurate gradient boosting
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1
        )
        
        # CatBoost - Robust gradient boosting with great defaults
        cat_model = CatBoostClassifier(
            iterations=200,
            depth=8,
            learning_rate=0.1,
            l2_leaf_reg=3,
            class_weights=None,  # Will auto-balance
            random_state=self.random_state,
            thread_count=-1,
            verbose=False
        )
        
        # Random Forest - Reliable ensemble method
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features=0.7,
            class_weight='balanced',
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Extra Trees - More randomized ensemble for diversity
        et_model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features=0.7,
            class_weight='balanced',
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Store base models
        self.base_models = {
            'xgboost': xgb_model,
            'lightgbm': lgb_model,
            'catboost': cat_model,
            'random_forest': rf_model,
            'extra_trees': et_model
        }
        
        # Meta-learner - XGBoost as meta-learner for final predictions
        meta_learner = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
            eval_metric='mlogloss'
        )
        
        # Create stacking ensemble
        print("\n🏗️ Building stacking ensemble...")
        self.ensemble_model = StackingClassifier(
            estimators=list(self.base_models.items()),
            final_estimator=meta_learner,
            cv=5,  # 5-fold cross-validation for meta-features
            stack_method='predict_proba',  # Use probabilities for meta-features
            n_jobs=-1,
            passthrough=False  # Don't pass original features to meta-learner
        )
        
        # Train the stacking ensemble
        print("\n🚀 Training stacking ensemble (this may take a while)...")
        start_time = time.time()
        
        try:
            self.ensemble_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            print(f"✅ Stacking ensemble training completed in {training_time:.1f}s")
            
            # Also train individual models for comparison and fallback
            print("\n🔄 Training individual models for analysis...")
            individual_times = {}
            for name, model in self.base_models.items():
                start_individual = time.time()
                model.fit(X_train, y_train)
                individual_times[name] = time.time() - start_individual
                print(f"  ✅ {name}: {individual_times[name]:.1f}s")
            
            # Set main model to the ensemble
            self.model = self.ensemble_model
            
        except Exception as e:
            print(f"❌ Error training stacking ensemble: {e}")
            print("🔄 Falling back to voting ensemble...")
            
            # Fallback to voting ensemble if stacking fails
            self.ensemble_model = VotingClassifier(
                estimators=list(self.base_models.items()),
                voting='soft',  # Use probabilities for voting
                n_jobs=-1
            )
            
            self.ensemble_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            print(f"✅ Voting ensemble training completed in {training_time:.1f}s")
            self.model = self.ensemble_model
        
        # Enable ensemble mode
        self.ensemble_mode = True
        
        print(f"\n🎯 Advanced ensemble model training completed!")
        print(f"📊 Total training time: {training_time:.1f}s")
        print(f"🧠 Using {len(self.base_models)} base models with meta-learner")
        print(f"🎯 Optimized for {len(self.minority_classes)} minority classes")
    
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Make predictions
        if self.ensemble_mode and self.ensemble_model is not None:
            # Ensemble prediction
            y_pred = self.ensemble_model.predict(X_test)
            print(f"🚀 Using advanced stacking ensemble prediction")
        else:
            # Single model prediction
            y_pred = self.model.predict(X_test)
            print(f"🌳 Using single Random Forest prediction")
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        target_names = [self.label_mapping[i] for i in sorted(self.label_mapping.keys())]
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion Matrix Visualization
        print("\n🔍 Generating Confusion Matrix Visualizations...")
        self.plot_confusion_matrix(y_test, y_pred)
        
        # Feature importance
        self.plot_feature_importance()
        
        return accuracy, y_pred
    
    def predict_ensemble(self, X_test):
        """Make ensemble predictions using advanced stacking ensemble"""
        if not self.ensemble_mode or self.ensemble_model is None:
            return self.model.predict(X_test)
        
        # Use the stacking ensemble for prediction
        return self.ensemble_model.predict(X_test)
    
    def get_ensemble_predictions_breakdown(self, X_test):
        """Get individual predictions from each base model for analysis"""
        if not self.ensemble_mode or not self.base_models:
            return None
        
        predictions = {}
        probabilities = {}
        
        for name, model in self.base_models.items():
            try:
                predictions[name] = model.predict(X_test)
                probabilities[name] = model.predict_proba(X_test)
            except Exception as e:
                print(f"⚠️ Error getting predictions from {name}: {e}")
                predictions[name] = None
                probabilities[name] = None
        
        # Also get ensemble prediction
        if self.ensemble_model is not None:
            predictions['ensemble'] = self.ensemble_model.predict(X_test)
            probabilities['ensemble'] = self.ensemble_model.predict_proba(X_test)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def plot_feature_importance(self, top_n=16):
        """Plot feature importance"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Get feature importance
        feature_importance = self.model.feature_importances_
        
        # Use stored feature names or create generic names if not available
        if hasattr(self, 'feature_names') and self.feature_names:
            feature_names = self.feature_names
        else:
            # Fallback to generic feature names
            feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
            print("⚠️ Using generic feature names. Feature names not stored during data preparation.")
        
        # Ensure feature names and importance arrays have the same length
        if len(feature_names) != len(feature_importance):
            print(f"⚠️ Mismatch: {len(feature_names)} feature names vs {len(feature_importance)} importance values")
            # Truncate or pad as needed
            min_len = min(len(feature_names), len(feature_importance))
            feature_names = feature_names[:min_len]
            feature_importance = feature_importance[:min_len]
        
        # Create DataFrame for plotting
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_df.head(top_n), x='Importance', y='Feature')
        plt.title(f'Top {top_n} Feature Importance - Random Forest')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_df
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        print(f"🔄 Running {cv}-fold cross-validation...")
        
        # Cross-validation with progress bar
        scores = []
        with tqdm(total=cv, desc="Cross-Validation", unit="fold") as pbar:
            for i in range(cv):
                # Get individual fold score
                fold_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
                if i == 0:  # Only calculate once, but show progress
                    scores = fold_scores
                pbar.update(1)
                time.sleep(0.1)  # Small delay to show progress
        
        print(f"📊 Cross-validation scores: {scores}")
        print(f"🎯 Mean CV accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        # Scale features if scaler was used
        if hasattr(self, 'scaler') and hasattr(self.scaler, 'scale_'):
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
        else:
            predictions = self.model.predict(X)
        
        # Convert predictions to chemical names
        predicted_chemicals = [self.label_mapping[pred] for pred in predictions]
        
        return predictions, predicted_chemicals
    
    def save_model(self, model_path="models/random_forest_model.pkl", scaler_path="models/scaler.pkl"):
        """Save the trained model and scaler"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Create models directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save main model and scaler
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        
        # Save ensemble components if in ensemble mode
        if self.ensemble_mode:
            # Save stacking ensemble model
            if self.ensemble_model is not None:
                ensemble_path = model_path.replace(".pkl", "_stacking_ensemble.pkl")
                joblib.dump(self.ensemble_model, ensemble_path)
                print(f"Stacking ensemble saved to {ensemble_path}")
            
            # Save individual base models
            if self.base_models:
                base_models_path = model_path.replace(".pkl", "_base_models.pkl")
                joblib.dump(self.base_models, base_models_path)
                print(f"Base models saved to {base_models_path}")
            
            # Save ensemble metadata
            ensemble_metadata = {
                'ensemble_mode': self.ensemble_mode,
                'minority_classes': self.minority_classes,
                'majority_classes': self.majority_classes,
                'model_type': 'stacking_ensemble'
            }
            metadata_path = model_path.replace(".pkl", "_ensemble_metadata.pkl")
            joblib.dump(ensemble_metadata, metadata_path)
            print(f"Ensemble metadata saved to {metadata_path}")
    
    def load_model(self, model_path="models/random_forest_model.pkl", scaler_path="models/scaler.pkl"):
        """Load a pre-trained model and scaler"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("Model and scaler loaded successfully!")
            
            # Try to load ensemble components if they exist
            metadata_path = model_path.replace(".pkl", "_ensemble_metadata.pkl")
            ensemble_path = model_path.replace(".pkl", "_stacking_ensemble.pkl")
            base_models_path = model_path.replace(".pkl", "_base_models.pkl")
            
            if os.path.exists(metadata_path):
                # Load ensemble metadata
                ensemble_metadata = joblib.load(metadata_path)
                self.ensemble_mode = ensemble_metadata['ensemble_mode']
                self.minority_classes = ensemble_metadata['minority_classes']
                self.majority_classes = ensemble_metadata['majority_classes']
                model_type = ensemble_metadata.get('model_type', 'simple_ensemble')
                print("Ensemble metadata loaded successfully!")
                
                # Load stacking ensemble if it exists
                if os.path.exists(ensemble_path):
                    self.ensemble_model = joblib.load(ensemble_path)
                    print("Stacking ensemble loaded successfully!")
                    
                    # Load base models if they exist
                    if os.path.exists(base_models_path):
                        self.base_models = joblib.load(base_models_path)
                        print(f"Base models loaded successfully! ({len(self.base_models)} models)")
                    
                    print(f"🚀 Advanced ensemble mode enabled with {len(self.minority_classes)} minority classes")
                    print(f"🧠 Model type: {model_type}")
                else:
                    print("⚠️ Ensemble metadata found but stacking ensemble missing")
                    # Try to load legacy logistic model for backward compatibility
                    lr_path = model_path.replace(".pkl", "_logistic.pkl")
                    if os.path.exists(lr_path):
                        print("🔄 Loading legacy ensemble format...")
                        # Reset to avoid conflicts
                        self.ensemble_mode = False
                        self.ensemble_model = None
                        self.base_models = {}
            else:
                # Reset ensemble mode for single models
                self.ensemble_mode = False
                self.minority_classes = []
                self.majority_classes = []
                self.ensemble_model = None
                self.base_models = {}
                
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")

    def balance_classes_comprehensive(self, X_train, y_train, balance_method='smote', downscale_majority=True, performance_based=True, target_classes=None):
        """
        Bilancia tutte le classi usando oversampling per le minoritarie e undersampling per le maggioritarie.
        Ora con bilanciamento basato sulle performance per ottimizzare i risultati.
        
        Args:
            X_train: Features di training
            y_train: Labels di training
            balance_method: Metodo di bilanciamento ('smote', 'adasyn', 'borderline', 'smote_tomek', 'smote_enn', 'aggressive')
            downscale_majority: Se applicare undersampling alle classi maggioritarie
            performance_based: Se usare target basati sulle performance delle classi
            target_classes: Lista specifica di classi da bilanciare (es. [7, 8, 18] per BIOETHANOL, BUTANE, METHANE)
        """
        
        # Se sono specificate classi target, usa il metodo originale
        if target_classes is not None:
            return self._balance_target_classes_only(X_train, y_train, target_classes, balance_method, downscale_majority)
        
        # Se è richiesto il bilanciamento aggressivo, usa il metodo dedicato
        if balance_method == 'aggressive':
            return self._balance_aggressive(X_train, y_train, downscale_majority)
        
        print("\n🎯 === BILANCIAMENTO PERFORMANCE-BASED DELLE CLASSI ===")
        
        # Definisci target basati sulle performance osservate dal classification report
        if performance_based:
            # Classi con performance scarse che necessitano più samples
            poor_performance_classes = {
                7: 0.06,   # BIOETHANOL (f1=0.51) -> 6% target
                18: 0.05,  # METHANE (f1=0.65) -> 5% target  
                23: 0.05,  # SODIUM_HYDROXIDE (f1=0.66) -> 5% target
                12: 0.04,  # FORMIC_ACID (f1=0.74) -> 4% target
                8: 0.04,   # BUTANE (f1=0.75) -> 4% target
            }
            
            # Classi minoritarie con performance discrete
            moderate_performance_classes = {
                6: 0.03,   # BALSAMIC_VINEGAR (f1=0.81) -> 3%
                17: 0.03,  # LIGHTER_FLUID (f1=0.80) -> 3%
                20: 0.03,  # PHOSPHORIC_ACID (f1=0.82) -> 3%
                5: 0.025,  # APPLE_VINEGAR (f1=0.87) -> 2.5%
                21: 0.025, # RED_WINE (f1=0.89) -> 2.5%
                14: 0.02,  # HYDROGEN_PEROXIDE (f1=0.96) -> 2% (già buone performance)
            }
            
            # Classi sovrarappresentate da ridurre drasticamente
            overrepresented_targets = {
                10: 0.03,  # DIESEL (f1=0.95, 16.46% -> 3%)
                16: 0.035, # KEROSENE (f1=0.97, 9.18% -> 3.5%)
                13: 0.04,  # GASOLINE (f1=0.95, 7.23% -> 4%)
                11: 0.045, # ETHANOL (f1=0.86, 8.08% -> 4.5%)
                0: 0.045,  # ACETIC_ACID (f1=0.94, 8.09% -> 4.5%)
            }
        
        # Analizza distribuzione iniziale
        print("📊 Distribuzione PRIMA del bilanciamento:")
        initial_counts = pd.Series(y_train).value_counts().sort_index()
        total_before = len(y_train)
        
        # Classifica le classi in base ai target definiti
        upsampling_targets = {}
        downsampling_targets = {}
        
        print(f"\n📊 Analisi distribuzione classi con target performance-based:")
        for class_id, count in initial_counts.items():
            percentage = (count / total_before) * 100
            class_name = self.label_mapping[class_id]
            
            # Determina target e azione
            if performance_based:
                if class_id in poor_performance_classes:
                    target_pct = poor_performance_classes[class_id] * 100
                    target_count = int(total_before * poor_performance_classes[class_id])
                    if count < target_count:
                        upsampling_targets[class_id] = target_count
                        status = f"📈 UPSAMPLE TO {target_pct:.1f}% (poor performance)"
                    else:
                        status = f"⚖️ ADEQUATE ({target_pct:.1f}% target)"
                        
                elif class_id in moderate_performance_classes:
                    target_pct = moderate_performance_classes[class_id] * 100
                    target_count = int(total_before * moderate_performance_classes[class_id])
                    if count < target_count:
                        upsampling_targets[class_id] = target_count
                        status = f"📈 UPSAMPLE TO {target_pct:.1f}% (moderate performance)"
                    else:
                        status = f"⚖️ ADEQUATE ({target_pct:.1f}% target)"
                        
                elif class_id in overrepresented_targets:
                    target_pct = overrepresented_targets[class_id] * 100
                    target_count = int(total_before * overrepresented_targets[class_id])
                    if count > target_count:
                        downsampling_targets[class_id] = target_count
                        status = f"📉 DOWNSAMPLE TO {target_pct:.1f}% (overrepresented)"
                    else:
                        status = f"⚖️ BALANCED ({target_pct:.1f}% target)"
                else:
                    status = "⚖️ BALANCED (no target change)"
            
            print(f"  {class_name}: {count:,} samples ({percentage:.2f}%) - {status}")
        
        print(f"\n🔍 Piano di bilanciamento:")
        print(f"  📈 Classi da aumentare: {len(upsampling_targets)}")
        for class_id in upsampling_targets:
            current = initial_counts[class_id]
            target = upsampling_targets[class_id]
            print(f"    - {self.label_mapping[class_id]}: {current:,} → {target:,} (+{target-current:,})")
            
        print(f"  📉 Classi da ridurre: {len(downsampling_targets)}")
        for class_id in downsampling_targets:
            current = initial_counts[class_id]
            target = downsampling_targets[class_id]
            print(f"    - {self.label_mapping[class_id]}: {current:,} → {target:,} ({target-current:,})")
        
        X_balanced = X_train.copy()
        y_balanced = y_train.copy()
        
        # Step 1: Downsampling
        if downscale_majority and downsampling_targets:
            print(f"\n📉 STEP 1: Downsampling classi sovrarappresentate...")
        
            if downsampling_targets:
                try:
                    undersampler = RandomUnderSampler(
                        sampling_strategy=downsampling_targets,
                        random_state=self.random_state
                    )
                    X_balanced, y_balanced = undersampler.fit_resample(X_balanced, y_balanced)
                    print(f"  ✅ Undersampling completato per {len(downsampling_targets)} classi!")
                    
                except Exception as e:
                    print(f"  ❌ Errore nell'undersampling: {e}")
                    print(f"  🔄 Procedendo senza undersampling...")
        
        # Step 2: Upsampling
        print(f"\n📈 STEP 2: Oversampling classi con performance scarse...")
        
        # Ricalcola conteggi dopo downsampling
        current_counts = pd.Series(y_balanced).value_counts().sort_index()
        current_total = len(y_balanced)
        
        # Aggiorna i target di upsampling basati sul nuovo totale
        adjusted_upsampling_targets = {}
        for class_id, original_target in upsampling_targets.items():
            # Mantieni la stessa proporzione rispetto al nuovo totale
            if class_id in poor_performance_classes:
                adjusted_target = int(current_total * poor_performance_classes[class_id])
            elif class_id in moderate_performance_classes:
                adjusted_target = int(current_total * moderate_performance_classes[class_id])
            else:
                continue
                
            current_count = current_counts.get(class_id, 0)
            if current_count < adjusted_target:
                adjusted_upsampling_targets[class_id] = adjusted_target
                increase = adjusted_target - current_count
                performance_note = "POOR PERF" if class_id in poor_performance_classes else "MOD PERF"
                print(f"  📈 {self.label_mapping[class_id]}: {current_count:,} → {adjusted_target:,} (+{increase:,}) [{performance_note}]")
        
        if adjusted_upsampling_targets:
            try:
                if balance_method == 'smote':
                    oversampler = SMOTE(
                        sampling_strategy=adjusted_upsampling_targets,
                        random_state=self.random_state,
                        k_neighbors=3
                    )
                elif balance_method == 'adasyn':
                    oversampler = ADASYN(
                        sampling_strategy=adjusted_upsampling_targets,
                        random_state=self.random_state,
                        n_neighbors=3
                    )
                elif balance_method == 'borderline':
                    oversampler = BorderlineSMOTE(
                        sampling_strategy=adjusted_upsampling_targets,
                        random_state=self.random_state,
                        k_neighbors=3
                    )
                elif balance_method == 'smote_tomek':
                    oversampler = SMOTETomek(
                        sampling_strategy=adjusted_upsampling_targets,
                        random_state=self.random_state,
                        smote_kwargs={'k_neighbors': 3}
                    )
                elif balance_method == 'smote_enn':
                    oversampler = SMOTEENN(
                        sampling_strategy=adjusted_upsampling_targets,
                        random_state=self.random_state,
                        smote_kwargs={'k_neighbors': 3}
                    )
                else:
                    raise ValueError(f"Metodo di bilanciamento non supportato: {balance_method}")
                
                X_balanced, y_balanced = oversampler.fit_resample(X_balanced, y_balanced)
                print(f"  ✅ Oversampling completato per {len(adjusted_upsampling_targets)} classi!")
                
            except Exception as e:
                print(f"  ❌ Errore nell'oversampling: {e}")
                print(f"  🔄 Provo con parametri ridotti...")
                
                try:
                    oversampler = SMOTE(
                        sampling_strategy=adjusted_upsampling_targets,
                        random_state=self.random_state,
                        k_neighbors=1
                    )
                    X_balanced, y_balanced = oversampler.fit_resample(X_balanced, y_balanced)
                    print(f"  ✅ Oversampling completato con parametri ridotti per {len(adjusted_upsampling_targets)} classi!")
                
                except Exception as e2:
                    print(f"  ❌ Errore anche con parametri ridotti: {e2}")
                    print(f"  🔄 Procedendo senza oversampling...")
            else:
                print(f"  ℹ️ Nessuna classe da aumentare identificata.")
            
        # Analizza distribuzione finale
        print(f"\n📊 DISTRIBUZIONE FINALE:")
        final_counts = pd.Series(y_balanced).value_counts().sort_index()
        total_after = len(y_balanced)
        
        # Mostra tutte le classi che sono state modificate
        changed_classes = set(list(upsampling_targets.keys()) + list(downsampling_targets.keys()))
        
        print(f"\n📈 Classi modificate con target performance-based:")
        for class_id in sorted(changed_classes):
            if class_id in final_counts.index:
                count = final_counts[class_id]
                percentage = (count / total_after) * 100
                class_name = self.label_mapping[class_id]
                initial_count = initial_counts.get(class_id, 0)
                change = count - initial_count
                
                # Identifica il tipo di modifica
                if class_id in upsampling_targets:
                    if class_id in poor_performance_classes:
                        modification_type = "📈 UPSAMPLED (POOR PERF)"
                        target_f1 = "f1<0.70"
                    else:
                        modification_type = "📈 UPSAMPLED (MOD PERF)"
                        target_f1 = "f1≥0.70"
                elif class_id in downsampling_targets:
                    modification_type = "📉 DOWNSAMPLED (OVERREP)"
                    target_f1 = "f1≥0.90"
                else:
                    modification_type = "⚖️ UNCHANGED"
                    target_f1 = ""
                
                if change > 0:
                    direction = f"[+{change:,}]"
                elif change < 0:
                    direction = f"[{change:,}]"
                else:
                    direction = "[=]"
                
                print(f"  {modification_type}: {class_name}")
                print(f"    📊 {count:,} samples ({percentage:.2f}%) {direction} {target_f1}")
        
        # Mostra statistiche globali migliorative
        print(f"\n📊 Statistiche finali performance-based:")
        final_percentages = [(final_counts[i] / total_after) * 100 for i in final_counts.index]
        mean_percentage = np.mean(final_percentages)
        std_percentage = np.std(final_percentages)
        min_percentage = np.min(final_percentages)
        max_percentage = np.max(final_percentages)
        
        # Calcola miglioramenti specifici per le classi con performance scarse
        poor_perf_improvements = []
        for class_id in poor_performance_classes.keys():
            if class_id in final_counts.index and class_id in initial_counts.index:
                initial_pct = (initial_counts[class_id] / total_before) * 100
                final_pct = (final_counts[class_id] / total_after) * 100
                improvement = final_pct / initial_pct if initial_pct > 0 else 0
                poor_perf_improvements.append(improvement)
                print(f"  🎯 {self.label_mapping[class_id]}: {initial_pct:.2f}% → {final_pct:.2f}% ({improvement:.1f}x)")
            
        avg_poor_improvement = np.mean(poor_perf_improvements) if poor_perf_improvements else 1.0
        
        print(f"\n  📊 Distribuzione percentuale globale:")
        print(f"    - Media: {mean_percentage:.2f}%")
        print(f"    - Deviazione standard: {std_percentage:.2f}%")
        print(f"    - Min: {min_percentage:.2f}%")
        print(f"    - Max: {max_percentage:.2f}%")
        print(f"    - Rapporto Max/Min: {max_percentage/min_percentage:.1f}x")
        print(f"    - Miglioramento medio classi scarse: {avg_poor_improvement:.1f}x")
        
        print(f"\n✅ Bilanciamento performance-based completato!")
        print(f"📈 Dataset size: {total_before:,} → {total_after:,} samples ({total_after - total_before:+,})")
        print(f"🎯 Classi bilanciate: {len(changed_classes)} su {len(initial_counts)} totali")
        print(f"🚀 Focus su {len(poor_performance_classes)} classi con performance scarse")
        
        return X_balanced, y_balanced
    
    def _balance_aggressive(self, X_train, y_train, downscale_majority=True):
        """
        Bilanciamento aggressivo multistadio che combina più tecniche per massimizzare le performance.
        
        Args:
            X_train: Features di training
            y_train: Labels di training
            downscale_majority: Se applicare undersampling alle classi maggioritarie
        """
        
        print("\n🚀 === BILANCIAMENTO AGGRESSIVO MULTISTADIO ===")
        print("🎯 Applicando combinazione di tecniche per bilanciamento estremo")
        
        # Analizza distribuzione iniziale
        print("📊 Distribuzione PRIMA del bilanciamento aggressivo:")
        initial_counts = pd.Series(y_train).value_counts().sort_index()
        total_before = len(y_train)
        
        # Definisci target aggressivi basati su analisi di performance
        extremely_poor_classes = {
            7: 0.08,   # BIOETHANOL (f1=0.51) -> 8% target (molto aggressivo)
            18: 0.07,  # METHANE (f1=0.65) -> 7% target
            23: 0.06,  # SODIUM_HYDROXIDE (f1=0.66) -> 6% target
            12: 0.05,  # FORMIC_ACID (f1=0.74) -> 5% target
            8: 0.05,   # BUTANE (f1=0.75) -> 5% target
        }
        
        # Classi con performance moderate ma che possono beneficiare di boost
        moderate_classes = {
            6: 0.045,  # BALSAMIC_VINEGAR (f1=0.81) -> 4.5%
            17: 0.04,  # LIGHTER_FLUID (f1=0.80) -> 4%
            20: 0.04,  # PHOSPHORIC_ACID (f1=0.82) -> 4%
            5: 0.035,  # APPLE_VINEGAR (f1=0.87) -> 3.5%
            21: 0.035, # RED_WINE (f1=0.89) -> 3.5%
            14: 0.03,  # HYDROGEN_PEROXIDE (f1=0.96) -> 3%
            19: 0.03,  # NITROMETHANE -> 3%
            9: 0.025,  # CALCIUM_NITRATE -> 2.5%
            22: 0.025, # SODIUM_HYDROXIDE -> 2.5%
            24: 0.025, # WATER_VAPOR -> 2.5%
        }
        
        # Classi sovrarappresentate da ridurre drasticamente
        overrepresented_classes = {
            10: 0.02,  # DIESEL (f1=0.95, 16.46% -> 2%) - riduzione drastica
            16: 0.025, # KEROSENE (f1=0.97, 9.18% -> 2.5%)
            13: 0.025, # GASOLINE (f1=0.95, 7.23% -> 2.5%)
            11: 0.03,  # ETHANOL (f1=0.86, 8.08% -> 3%)
            0: 0.03,   # ACETIC_ACID (f1=0.94, 8.09% -> 3%)
        }
        
        print(f"\n🎯 Piano di bilanciamento aggressivo:")
        print(f"  🚀 Classi estremamente povere (boost alto): {len(extremely_poor_classes)}")
        print(f"  📈 Classi moderate (boost medio): {len(moderate_classes)}")
        print(f"  📉 Classi sovrarappresentate (riduzione drastica): {len(overrepresented_classes)}")
        
        # Mostra dettagli per ogni classe
        for class_id, count in initial_counts.items():
            percentage = (count / total_before) * 100
            class_name = self.label_mapping[class_id]
            
            if class_id in extremely_poor_classes:
                target_pct = extremely_poor_classes[class_id] * 100
                multiplier = extremely_poor_classes[class_id] / (percentage / 100)
                status = f"🚀 BOOST ESTREMO to {target_pct:.1f}% ({multiplier:.1f}x)"
            elif class_id in moderate_classes:
                target_pct = moderate_classes[class_id] * 100
                multiplier = moderate_classes[class_id] / (percentage / 100)
                status = f"📈 BOOST MEDIO to {target_pct:.1f}% ({multiplier:.1f}x)"
            elif class_id in overrepresented_classes:
                target_pct = overrepresented_classes[class_id] * 100
                reduction = (percentage / 100) / overrepresented_classes[class_id]
                status = f"📉 RIDUZIONE DRASTICA to {target_pct:.1f}% (-{reduction:.1f}x)"
            else:
                status = "⚖️ MANTENIMENTO"
            
            print(f"  {class_name}: {count} ({percentage:.2f}%) - {status}")
        
        X_balanced = X_train.copy()
        y_balanced = y_train.copy()
        
        # FASE 1: Undersampling aggressivo delle classi sovrarappresentate
        if downscale_majority and overrepresented_classes:
            print(f"\n📉 FASE 1: Undersampling aggressivo classi sovrarappresentate...")
            
            downsampling_targets = {}
            for class_id, target_ratio in overrepresented_classes.items():
                current_count = initial_counts.get(class_id, 0)
                target_count = int(total_before * target_ratio)
                if current_count > target_count:
                    downsampling_targets[class_id] = target_count
                    reduction = current_count - target_count
                    print(f"  📉 {self.label_mapping[class_id]}: {current_count} → {target_count} (-{reduction})")
            
            if downsampling_targets:
                try:
                    undersampler = RandomUnderSampler(
                        sampling_strategy=downsampling_targets,
                        random_state=self.random_state
                    )
                    X_balanced, y_balanced = undersampler.fit_resample(X_balanced, y_balanced)
                    print(f"  ✅ Undersampling aggressivo completato per {len(downsampling_targets)} classi!")
                except Exception as e:
                    print(f"  ❌ Errore nell'undersampling aggressivo: {e}")
        
        # FASE 2: Oversampling multistadio - prima le classi estremamente povere
        print(f"\n🚀 FASE 2A: Oversampling estremo per classi con performance pessime...")
        
        current_counts = pd.Series(y_balanced).value_counts().sort_index()
        current_total = len(y_balanced)
        
        # Oversampling per classi estremamente povere
        extreme_oversampling_targets = {}
        for class_id, target_ratio in extremely_poor_classes.items():
            current_count = current_counts.get(class_id, 0)
            target_count = int(current_total * target_ratio)
            if current_count < target_count:
                extreme_oversampling_targets[class_id] = target_count
                increase = target_count - current_count
                print(f"  🚀 {self.label_mapping[class_id]}: {current_count} → {target_count} (+{increase})")
        
        if extreme_oversampling_targets:
            try:
                # Usa BorderlineSMOTE con parametri aggressivi per le classi più difficili
                oversampler = BorderlineSMOTE(
                    sampling_strategy=extreme_oversampling_targets,
                    random_state=self.random_state,
                    k_neighbors=5,  # Più vicini per migliore interpolazione
                    m_neighbors=10,  # Più vicini per identificare borderline
                    kind='borderline-1'  # Tipo più aggressivo
                )
                X_balanced, y_balanced = oversampler.fit_resample(X_balanced, y_balanced)
                print(f"  ✅ Oversampling estremo completato per {len(extreme_oversampling_targets)} classi!")
            except Exception as e:
                print(f"  ❌ Errore nell'oversampling estremo: {e}")
                # Fallback a SMOTE normale
                try:
                    oversampler = SMOTE(
                        sampling_strategy=extreme_oversampling_targets,
                        random_state=self.random_state,
                        k_neighbors=3
                    )
                    X_balanced, y_balanced = oversampler.fit_resample(X_balanced, y_balanced)
                    print(f"  ✅ Oversampling estremo completato con SMOTE fallback!")
                except Exception as e2:
                    print(f"  ❌ Errore anche con SMOTE fallback: {e2}")
        
        # FASE 2B: Oversampling per classi moderate
        print(f"\n📈 FASE 2B: Oversampling per classi con performance moderate...")
        
        current_counts = pd.Series(y_balanced).value_counts().sort_index()
        current_total = len(y_balanced)
        
        moderate_oversampling_targets = {}
        for class_id, target_ratio in moderate_classes.items():
            current_count = current_counts.get(class_id, 0)
            target_count = int(current_total * target_ratio)
            if current_count < target_count:
                moderate_oversampling_targets[class_id] = target_count
                increase = target_count - current_count
                print(f"  📈 {self.label_mapping[class_id]}: {current_count} → {target_count} (+{increase})")
        
        if moderate_oversampling_targets:
            try:
                # Usa ADASYN per adattarsi alla difficoltà delle classi moderate
                oversampler = ADASYN(
                    sampling_strategy=moderate_oversampling_targets,
                    random_state=self.random_state,
                    n_neighbors=5
                )
                X_balanced, y_balanced = oversampler.fit_resample(X_balanced, y_balanced)
                print(f"  ✅ Oversampling moderato completato per {len(moderate_oversampling_targets)} classi!")
            except Exception as e:
                print(f"  ❌ Errore nell'oversampling moderato: {e}")
                # Fallback a SMOTE
                try:
                    oversampler = SMOTE(
                        sampling_strategy=moderate_oversampling_targets,
                        random_state=self.random_state,
                        k_neighbors=3
                    )
                    X_balanced, y_balanced = oversampler.fit_resample(X_balanced, y_balanced)
                    print(f"  ✅ Oversampling moderato completato con SMOTE fallback!")
                except Exception as e2:
                    print(f"  ❌ Errore anche con SMOTE fallback: {e2}")
        
        # FASE 3: Pulizia finale con Tomek Links per rimuovere sovrapposizioni
        print(f"\n🧹 FASE 3: Pulizia finale con Tomek Links...")
        
        try:
            cleaner = TomekLinks()
            X_balanced, y_balanced = cleaner.fit_resample(X_balanced, y_balanced)
            print(f"  ✅ Pulizia Tomek Links completata!")
        except Exception as e:
            print(f"  ❌ Errore nella pulizia Tomek Links: {e}")
            print(f"  🔄 Procedendo senza pulizia finale...")
        
        # Analisi finale
        print(f"\n📊 DISTRIBUZIONE FINALE DOPO BILANCIAMENTO AGGRESSIVO:")
        final_counts = pd.Series(y_balanced).value_counts().sort_index()
        total_after = len(y_balanced)
        
        # Statistiche di miglioramento
        poor_performance_improvements = []
        overrepresented_reductions = []
        
        for class_id, count in final_counts.items():
            percentage = (count / total_after) * 100
            class_name = self.label_mapping[class_id]
            initial_count = initial_counts.get(class_id, 0)
            initial_pct = (initial_count / total_before) * 100
            change = count - initial_count
            
            # Calcola miglioramenti
            if class_id in extremely_poor_classes:
                improvement = percentage / initial_pct if initial_pct > 0 else 0
                poor_performance_improvements.append(improvement)
                status = f"🚀 BOOST ESTREMO ({improvement:.1f}x)"
            elif class_id in moderate_classes:
                improvement = percentage / initial_pct if initial_pct > 0 else 0
                poor_performance_improvements.append(improvement)
                status = f"📈 BOOST MEDIO ({improvement:.1f}x)"
            elif class_id in overrepresented_classes:
                reduction = initial_pct / percentage if percentage > 0 else 0
                overrepresented_reductions.append(reduction)
                status = f"📉 RIDOTTO ({reduction:.1f}x meno)"
            else:
                status = "⚖️ MANTENUTO"
            
            direction = f"[{change:+d}]" if change != 0 else "[=]"
            print(f"  {class_name}: {count} ({percentage:.2f}%) {direction} - {status}")
        
        # Statistiche globali
        print(f"\n🎯 STATISTICHE BILANCIAMENTO AGGRESSIVO:")
        final_percentages = [(final_counts[i] / total_after) * 100 for i in final_counts.index]
        mean_percentage = np.mean(final_percentages)
        std_percentage = np.std(final_percentages)
        min_percentage = np.min(final_percentages)
        max_percentage = np.max(final_percentages)
        
        avg_poor_improvement = np.mean(poor_performance_improvements) if poor_performance_improvements else 1.0
        avg_overrep_reduction = np.mean(overrepresented_reductions) if overrepresented_reductions else 1.0
        
        print(f"  📊 Distribuzione finale:")
        print(f"    - Media: {mean_percentage:.2f}%")
        print(f"    - Deviazione standard: {std_percentage:.2f}%")
        print(f"    - Range: {min_percentage:.2f}% - {max_percentage:.2f}%")
        print(f"    - Rapporto Max/Min: {max_percentage/min_percentage:.1f}x")
        print(f"  📈 Miglioramento medio classi povere: {avg_poor_improvement:.1f}x")
        print(f"  📉 Riduzione media classi sovrarappresentate: {avg_overrep_reduction:.1f}x")
        print(f"  📊 Dataset size: {total_before:,} → {total_after:,} ({total_after-total_before:+,})")
        
        print(f"\n🚀 BILANCIAMENTO AGGRESSIVO COMPLETATO!")
        print(f"🎯 Focus massimo su {len(extremely_poor_classes)} classi critiche")
        print(f"📈 Boost applicato a {len(moderate_classes)} classi moderate")
        print(f"📉 Riduzione applicata a {len(overrepresented_classes)} classi sovrarappresentate")
        print(f"🧹 Pulizia finale applicata per ridurre sovrapposizioni")
        
        return X_balanced, y_balanced
            
    def _balance_target_classes_only(self, X_train, y_train, target_classes=[7, 8, 18], balance_method='smote', downscale_majority=True):
        """
        Bilancia solo le classi specificate usando il metodo originale.
        
        Args:
            X_train: Features di training
            y_train: Labels di training
            target_classes: Lista delle classi minoritarie da bilanciare [7=BIOETHANOL, 8=BUTANE, 18=METHANE]
            balance_method: Metodo di bilanciamento ('smote', 'adasyn', 'borderline', 'smote_tomek', 'smote_enn')
            downscale_majority: Se applicare undersampling alle classi maggioritarie
        """
        print("\n🎯 === BILANCIAMENTO CLASSI SPECIFICHE ===")
        print(f"🎯 Target classes: {[self.label_mapping[c] for c in target_classes]}")
        
        # Analizza distribuzione iniziale
        print("📊 Distribuzione PRIMA del bilanciamento:")
        initial_counts = pd.Series(y_train).value_counts().sort_index()
        total_before = len(y_train)
        
        # Identifica classi maggioritarie (sopra il 10% del dataset)
        majority_threshold = 0.10  # 10% del dataset
        majority_classes = []
        minority_classes = target_classes
        
        for class_id, count in initial_counts.items():
            percentage = (count / total_before) * 100
            class_name = self.label_mapping[class_id]
            print(f"  {class_name}: {count} samples ({percentage:.2f}%)")
            
            if percentage > majority_threshold * 100:
                majority_classes.append(class_id)
        
        print(f"\n🔍 Classi identificate:")
        print(f"  Maggioritarie (>{majority_threshold*100:.0f}%): {[self.label_mapping[c] for c in majority_classes]}")
        print(f"  Minoritarie target: {[self.label_mapping[c] for c in minority_classes]}")
        
        X_balanced = X_train.copy()
        y_balanced = y_train.copy()
        
        # Step 1: Downscale majority classes
        if downscale_majority and majority_classes:
            print(f"\n📉 STEP 1: Downscaling classi maggioritarie...")
            
            # Calcola target count per le classi maggioritarie (ridotto al 6% del dataset originale)
            majority_target_percentage = 0.06
            majority_target_count = int(total_before * majority_target_percentage)
            
            undersampling_strategy = {}
            for class_id in majority_classes:
                current_count = initial_counts.get(class_id, 0)
                if current_count > majority_target_count:
                    undersampling_strategy[class_id] = majority_target_count
                    print(f"  📉 {self.label_mapping[class_id]}: {current_count} → {majority_target_count} samples")
            
            if undersampling_strategy:
                try:
                    # Usa Random Undersampling per semplicità e velocità
                    undersampler = RandomUnderSampler(
                        sampling_strategy=undersampling_strategy,
                        random_state=self.random_state
                    )
                    X_balanced, y_balanced = undersampler.fit_resample(X_balanced, y_balanced)
                    print(f"  ✅ Undersampling completato!")
                    
                except Exception as e:
                    print(f"  ❌ Errore nell'undersampling: {e}")
                    print(f"  🔄 Procedendo senza undersampling...")
        
        # Step 2: Oversample minority classes  
        print(f"\n📈 STEP 2: Oversampling classi minoritarie target...")
        
        # Calcola nuova distribuzione dopo undersampling
        current_counts = pd.Series(y_balanced).value_counts().sort_index()
        current_total = len(y_balanced)
        
        # Definisci strategia di oversampling per le classi minoritarie
        # Aumentiamo le classi minoritarie al 4% del dataset corrente
        minority_target_percentage = 0.04
        minority_target_count = int(current_total * minority_target_percentage)
        
        oversampling_strategy = {}
        for class_id in minority_classes:
            current_count = current_counts.get(class_id, 0)
            if current_count < minority_target_count:
                oversampling_strategy[class_id] = minority_target_count
                print(f"  📈 {self.label_mapping[class_id]}: {current_count} → {minority_target_count} samples")
        
        if oversampling_strategy:
            try:
                if balance_method == 'smote':
                    oversampler = SMOTE(
                        sampling_strategy=oversampling_strategy,
                        random_state=self.random_state,
                        k_neighbors=3
                    )
                elif balance_method == 'adasyn':
                    oversampler = ADASYN(
                        sampling_strategy=oversampling_strategy,
                        random_state=self.random_state,
                        n_neighbors=3
                    )
                elif balance_method == 'borderline':
                    oversampler = BorderlineSMOTE(
                        sampling_strategy=oversampling_strategy,
                        random_state=self.random_state,
                        k_neighbors=3
                    )
                elif balance_method == 'smote_tomek':
                    oversampler = SMOTETomek(
                        sampling_strategy=oversampling_strategy,
                        random_state=self.random_state,
                        smote_kwargs={'k_neighbors': 3}
                    )
                elif balance_method == 'smote_enn':
                    oversampler = SMOTEENN(
                        sampling_strategy=oversampling_strategy,
                    random_state=self.random_state,
                        smote_kwargs={'k_neighbors': 3}
                    )
                else:
                    raise ValueError(f"Metodo di bilanciamento non supportato: {balance_method}")
                
                X_balanced, y_balanced = oversampler.fit_resample(X_balanced, y_balanced)
                print(f"  ✅ Oversampling completato!")
                
            except Exception as e:
                print(f"  ❌ Errore nell'oversampling: {e}")
                print(f"  🔄 Provo con parametri ridotti...")
                
                try:
                    oversampler = SMOTE(
                        sampling_strategy=oversampling_strategy,
                        random_state=self.random_state,
                        k_neighbors=1
                    )
                    X_balanced, y_balanced = oversampler.fit_resample(X_balanced, y_balanced)
                    print(f"  ✅ Oversampling completato con parametri ridotti!")
                
                except Exception as e2:
                    print(f"  ❌ Errore anche con parametri ridotti: {e2}")
        
        # Analizza distribuzione finale
        print(f"\n📊 DISTRIBUZIONE FINALE:")
        final_counts = pd.Series(y_balanced).value_counts().sort_index()
        total_after = len(y_balanced)
        
        # Mostra le classi più importanti
        important_classes = majority_classes + minority_classes
        for class_id in sorted(important_classes):
            if class_id in final_counts.index:
                count = final_counts[class_id]
                percentage = (count / total_after) * 100
                class_name = self.label_mapping[class_id]
                initial_count = initial_counts.get(class_id, 0)
                change = count - initial_count
                direction = "+" if change > 0 else ""
                print(f"  {class_name}: {count} samples ({percentage:.2f}%) [{direction}{change}]")
        
        print(f"\n✅ Bilanciamento specifico completo!")
        print(f"📈 Dataset size: {total_before} → {total_after} samples ({total_after - total_before:+d})")
        
        return X_balanced, y_balanced

    def plot_confusion_matrix(self, y_test, y_pred, normalize_options=['counts', 'percentages'], save_path='visualizations/'):
        """
        Plot confusion matrix with multiple normalization options
        
        Args:
            y_test: True labels
            y_pred: Predicted labels  
            normalize_options: List of normalization types ('counts', 'percentages', 'true', 'pred')
            save_path: Directory to save the plots
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Get unique labels present in the data
        unique_labels = sorted(list(set(y_test) | set(y_pred)))
        target_names = [self.label_mapping[i] for i in unique_labels]
        
        # Create plots for each normalization option
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.ravel()
        
        normalization_configs = [
            ('counts', None, 'Confusion Matrix (Counts)'),
            ('percentages', 'true', 'Confusion Matrix (Percentages by True Class)'), 
            ('pred_percentages', 'pred', 'Confusion Matrix (Percentages by Predicted Class)'),
            ('overall_percentages', 'all', 'Confusion Matrix (Overall Percentages)')
        ]
        
        for idx, (name, normalize, title) in enumerate(normalization_configs):
            if name not in normalize_options and normalize is not None:
                continue
                
            # Calculate confusion matrix
            if normalize is None:
                cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
                fmt = 'd'
                cmap = 'Blues'
            elif normalize == 'true':
                cm = confusion_matrix(y_test, y_pred, labels=unique_labels, normalize='true') * 100
                fmt = '.1f'
                cmap = 'Blues'  
            elif normalize == 'pred':
                cm = confusion_matrix(y_test, y_pred, labels=unique_labels, normalize='pred') * 100
                fmt = '.1f'
                cmap = 'Greens'
            else:  # normalize == 'all'
                cm = confusion_matrix(y_test, y_pred, labels=unique_labels, normalize='all') * 100
                fmt = '.2f'
                cmap = 'Oranges'
            
            # Plot on subplot
            if idx < len(axes):
                ax = axes[idx]
                im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
                ax.figure.colorbar(im, ax=ax)
                
                # Add labels
                ax.set(xticks=np.arange(cm.shape[1]),
                       yticks=np.arange(cm.shape[0]),
                       xticklabels=target_names,
                       yticklabels=target_names,
                       title=title,
                       ylabel='True Label',
                       xlabel='Predicted Label')
                
                # Rotate the tick labels and set their alignment
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # Add text annotations
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        text_color = "white" if cm[i, j] > thresh else "black"
                        text = format(cm[i, j], fmt)
                        if normalize is not None and normalize != 'all':
                            text += '%'
                        ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=8)
        
        # Hide unused subplots
        for idx in range(len(normalization_configs), len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        
        # Save the comprehensive plot
        comprehensive_path = os.path.join(save_path, 'confusion_matrix_comprehensive.png')
        plt.savefig(comprehensive_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comprehensive confusion matrix saved to {comprehensive_path}")
        
        # Also create individual detailed plots
        self._save_individual_confusion_matrices(y_test, y_pred, unique_labels, target_names, save_path)
        
        plt.show()
        
        # Calculate and display key metrics from confusion matrix
        self._analyze_confusion_matrix_metrics(y_test, y_pred, unique_labels, target_names)
        
        return cm

    def _save_individual_confusion_matrices(self, y_test, y_pred, unique_labels, target_names, save_path):
        """Save individual confusion matrix plots for detailed analysis"""
        
        # 1. Counts matrix (large, detailed)
        plt.figure(figsize=(16, 12))
        cm_counts = confusion_matrix(y_test, y_pred, labels=unique_labels)
        sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names,
                   cbar_kws={'label': 'Number of Samples'})
        plt.title('Confusion Matrix - Sample Counts', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        counts_path = os.path.join(save_path, 'confusion_matrix_counts.png')
        plt.savefig(counts_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Normalized by true class (recall visualization)
        plt.figure(figsize=(16, 12))
        cm_norm_true = confusion_matrix(y_test, y_pred, labels=unique_labels, normalize='true')
        sns.heatmap(cm_norm_true, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names,
                   cbar_kws={'label': 'Recall (True Positive Rate)'})
        plt.title('Confusion Matrix - Normalized by True Class (Recall)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        recall_path = os.path.join(save_path, 'confusion_matrix_recall.png')
        plt.savefig(recall_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Normalized by predicted class (precision visualization)
        plt.figure(figsize=(16, 12))
        cm_norm_pred = confusion_matrix(y_test, y_pred, labels=unique_labels, normalize='pred')
        sns.heatmap(cm_norm_pred, annot=True, fmt='.2f', cmap='Greens',
                   xticklabels=target_names, yticklabels=target_names,
                   cbar_kws={'label': 'Precision (Positive Predictive Value)'})
        plt.title('Confusion Matrix - Normalized by Predicted Class (Precision)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        precision_path = os.path.join(save_path, 'confusion_matrix_precision.png')
        plt.savefig(precision_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Individual confusion matrices saved:")
        print(f"   📊 Counts: {counts_path}")
        print(f"   🎯 Recall: {recall_path}")
        print(f"   📈 Precision: {precision_path}")

    def _analyze_confusion_matrix_metrics(self, y_test, y_pred, unique_labels, target_names):
        """Analyze and display key metrics derived from confusion matrix"""
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        print(f"\n📊 CONFUSION MATRIX ANALYSIS")
        print("=" * 50)
        
        # Overall accuracy
        overall_accuracy = accuracy_score(y_test, y_pred)
        print(f"🎯 Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=unique_labels, average=None, zero_division=0
        )
        
        # Create a summary DataFrame
        metrics_df = pd.DataFrame({
            'Class': target_names,
            'Precision': precision,
            'Recall': recall, 
            'F1-Score': f1,
            'Support': support
        })
        
        # Identify problematic classes
        print(f"\n⚠️ CLASSES NEEDING ATTENTION:")
        problematic_classes = []
        
        for idx, row in metrics_df.iterrows():
            issues = []
            if row['Precision'] < 0.8:
                issues.append(f"Low Precision ({row['Precision']:.3f})")
            if row['Recall'] < 0.8:
                issues.append(f"Low Recall ({row['Recall']:.3f})")
            if row['F1-Score'] < 0.8:
                issues.append(f"Low F1 ({row['F1-Score']:.3f})")
            if row['Support'] < 10:
                issues.append(f"Low Support ({int(row['Support'])} samples)")
                
            if issues:
                problematic_classes.append((row['Class'], issues))
                print(f"   🔴 {row['Class']}: {', '.join(issues)}")
        
        if not problematic_classes:
            print("   ✅ All classes performing well!")
        
        # Calculate confusion matrix for error analysis
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        
        # Find most confused pairs
        print(f"\n🔀 MOST CONFUSED CLASS PAIRS:")
        confusion_pairs = []
        for i in range(len(unique_labels)):
            for j in range(len(unique_labels)):
                if i != j and cm[i][j] > 0:
                    confusion_pairs.append((
                        target_names[i], target_names[j], cm[i][j],
                        cm[i][j] / cm[i].sum() * 100  # Percentage of true class i predicted as j
                    ))
        
        # Sort by absolute count of misclassifications
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for i, (true_class, pred_class, count, percentage) in enumerate(confusion_pairs[:10]):
            print(f"   {i+1:2d}. {true_class} → {pred_class}: {count} cases ({percentage:.1f}%)")
        
        return metrics_df


def main():
    """Main function to train and evaluate the Random Forest model"""
    print("🚀 === Sensor Data Classification with Random Forest ===")
    
    # Initialize model
    print("🔧 Initializing Random Forest model...")
    rf_model = SensorRandomForestModel()
    
    # Load data
    print("📂 Loading preprocessed data...")
    df = rf_model.load_data()
    if df is None:
        return
    
    # Prepare data with advanced preprocessing and manual feature selection
    print("⚙️ Preparing data for training...")
    X_train, X_test, y_train, y_test = rf_model.prepare_data(
        df, 
        scale_features=True, 
        feature_engineering=True, 
        balance_classes=True, 
        balance_method='smote',
        manual_feature_selection=True,
        downscale_majority=True
    )
    print(f"📊 Training set size: {X_train.shape}")
    print(f"📊 Test set size: {X_test.shape}")
    
    # Train Random Forest model
    print("\n🏋️ Training Random Forest model...")
    rf_model.train_model(X_train, y_train)
    
    # Evaluate model
    print("\n📈 Evaluating model performance...")
    accuracy, y_pred = rf_model.evaluate_model(X_test, y_test)
    
    # Cross-validation
    print("\n🔄 === Cross-Validation ===")
    rf_model.cross_validate(X_train, y_train, cv=5)
    
    # Save model
    print("\n💾 Saving trained model...")
    rf_model.save_model()
    
    print("\n🎉 === Training Complete ===")
    print(f"🏆 Final Test Accuracy: {accuracy:.4f}")
    print("✅ Model ready for predictions!")


if __name__ == "__main__":
    main()
