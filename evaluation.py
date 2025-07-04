#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Sensor Random Forest Model
Allows interactive selection of:
- Feature Engineering (on/off)
- Feature Selection (manual/all features)
- Balancing methods
- Cross validation (on/off)
"""

from model import SensorRandomForestModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import os
import sys

class ModelEvaluator:
    def __init__(self):
        self.rf_model = SensorRandomForestModel()
        self.data = None
        self.results_history = []
        
    def load_data(self):
        """Load the dataset"""
        print("📂 Loading data...")
        self.data = self.rf_model.load_data()
        if self.data is None:
            print("❌ Failed to load data. Please ensure data/preprocessed_data.pkl exists.")
            return False
        print(f"✅ Data loaded successfully. Shape: {self.data.shape}")
        return True
    
    def display_main_menu(self):
        """Display the main menu options"""
        print("\n" + "="*60)
        print("🎯 SENSOR RANDOM FOREST MODEL EVALUATOR")
        print("="*60)
        print("1. 🔧 Configure and Run Single Evaluation")
        print("2. 📊 Compare Multiple Configurations")
        print("3. 📈 View Results History")
        print("4. 🔍 Analyze Class Distribution")
        print("5. 🚀 Advanced Stacking Ensemble (XGB+LGB+CAT+RF+ET)")
        print("6. ❌ Exit")
        print("="*60)
    
    def get_feature_engineering_choice(self):
        """Get user choice for feature engineering"""
        print("\n🔧 FEATURE ENGINEERING OPTIONS:")
        print("1. ✅ Use Feature Engineering (ratios, polynomials, statistics)")
        print("2. ❌ Skip Feature Engineering (use original features only)")
        
        while True:
            choice = input("Select option (1-2): ").strip()
            if choice == "1":
                return True
            elif choice == "2":
                return False
            else:
                print("❌ Invalid choice. Please select 1 or 2.")
    
    def get_feature_selection_choice(self):
        """Get user choice for feature selection"""
        print("\n🎯 FEATURE SELECTION OPTIONS:")
        print("1. 🎯 Use Manual Feature Selection (16 optimized features)")
        print("2. 📊 Keep All Features")
        
        while True:
            choice = input("Select option (1-2): ").strip()
            if choice == "1":
                return True
            elif choice == "2":
                return False
            else:
                print("❌ Invalid choice. Please select 1 or 2.")
    
    def get_balancing_choice(self):
        """Get user choice for class balancing"""
        print("\n⚖️ CLASS BALANCING OPTIONS:")
        print("1. ❌ No Balancing")
        print("2. 🎯 SMOTE (Synthetic Minority Oversampling)")
        print("3. 🔄 ADASYN (Adaptive Synthetic Sampling)")
        print("4. 🎯 BorderlineSMOTE")
        print("5. 🔄 SMOTE + Tomek Links")
        print("6. 🔄 SMOTE + Edited Nearest Neighbours")
        print("7. 🎯 Target Classes Only (3 least represented)")
        print("8. 🌍 Global Balancing (all classes)")
        print("9. 🚀 Aggressive Overall Balancing (Multi-stage aggressive)")
        
        balance_methods = {
            "1": (False, None),
            "2": (True, "smote"),
            "3": (True, "adasyn"),
            "4": (True, "borderline"),
            "5": (True, "smote_tomek"),
            "6": (True, "smote_enn"),
            "7": (True, "target_only"),
            "8": (True, "global"),
            "9": (True, "aggressive")
        }
        
        while True:
            choice = input("Select option (1-9): ").strip()
            if choice in balance_methods:
                return balance_methods[choice]
            else:
                print("❌ Invalid choice. Please select 1-9.")
    
    def get_cross_validation_choice(self):
        """Get user choice for cross validation"""
        print("\n🔄 CROSS VALIDATION OPTIONS:")
        print("1. ❌ No Cross Validation (train/test split only)")
        print("2. ✅ 5-Fold Cross Validation")
        print("3. ✅ 10-Fold Cross Validation")
        print("4. ✅ Custom K-Fold")
        
        while True:
            choice = input("Select option (1-4): ").strip()
            if choice == "1":
                return False, 0
            elif choice == "2":
                return True, 5
            elif choice == "3":
                return True, 10
            elif choice == "4":
                while True:
                    try:
                        k = int(input("Enter number of folds (2-20): "))
                        if 2 <= k <= 20:
                            return True, k
                        else:
                            print("❌ Please enter a value between 2 and 20.")
                    except ValueError:
                        print("❌ Please enter a valid number.")
            else:
                print("❌ Invalid choice. Please select 1-4.")
    
    def identify_least_represented_classes(self, n_classes=3):
        """Identify the least represented classes"""
        class_counts = self.data['LABEL'].value_counts().sort_values()
        least_represented = class_counts.head(n_classes)
        return least_represented.index.tolist()
    
    def perform_cross_validation(self, X, y, cv_folds=5):
        """Perform cross validation analysis"""
        print(f"\n🔄 Running {cv_folds}-fold cross validation...")
        
        # Stratified K-Fold to ensure balanced representation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.rf_model.random_state)
        
        cv_start = time.time()
        cv_scores = cross_val_score(self.rf_model.model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        cv_time = time.time() - cv_start
        
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        print(f"📊 CV Results: {mean_score:.4f} ± {std_score:.4f}")
        print(f"⏱️ CV Time: {cv_time:.1f} seconds")
        
        return {
            'scores': cv_scores,
            'mean': mean_score,
            'std': std_score,
            'cv_time': cv_time
        }
    
    def run_evaluation(self, config, save_model=False):
        """Run model evaluation with given configuration"""
        model_type = config.get('model_type', 'single')
        
        print(f"\n🚀 Running evaluation with configuration:")
        print(f"  🔧 Feature Engineering: {'✅' if config['feature_engineering'] else '❌'}")
        print(f"  🎯 Feature Selection: {'Manual (16 features)' if config['manual_selection'] else 'All features'}")
        print(f"  ⚖️ Balancing: {config['balance_method'] if config['balance_classes'] else 'None'}")
        cv_text = f"{config['cv_folds']}-fold" if config['use_cv'] else 'None'
        print(f"  🔄 Cross Validation: {cv_text}")
        
        if model_type == 'ensemble':
            print(f"  🚀 Model Type: Advanced Stacking Ensemble")
            print(f"  🧠 Base Models: XGBoost, LightGBM, CatBoost, RF, ExtraTrees")
            print(f"  🎯 Minority Threshold: {config['minority_threshold']:.2%}")
        else:
            print(f"  🌳 Model Type: Single Random Forest")
        
        start_time = time.time()
        
        # Prepare target classes for balancing if needed
        target_classes = None
        if config['balance_classes'] and config['balance_method'] == 'target_only':
            target_classes = self.identify_least_represented_classes()
            print(f"  🎯 Target classes: {[self.rf_model.label_mapping[c] for c in target_classes]}")
        
        # Prepare data
        try:
            X_train, X_test, y_train, y_test = self.rf_model.prepare_data(
                self.data,
                scale_features=True,
                feature_engineering=config['feature_engineering'],
                balance_classes=config['balance_classes'],
                balance_method=config['balance_method'] if config['balance_method'] != 'target_only' else 'smote',
                manual_feature_selection=config['manual_selection'],
                downscale_majority=(config['balance_method'] in ['global', 'aggressive']),
                target_classes=target_classes
            )
        except Exception as e:
            print(f"❌ Error in data preparation: {e}")
            return None
        
        prep_time = time.time() - start_time
        
        print(f"\n📊 Dataset sizes:")
        print(f"  📈 Training: {X_train.shape}")
        print(f"  📉 Testing: {X_test.shape}")
        print(f"⏱️ Preparation time: {prep_time:.1f}s")
        
        # Train model
        print("\n🏋️ Training model...")
        train_start = time.time()
        
        if model_type == 'ensemble':
            # Train ensemble model
            minority_threshold = config['minority_threshold']
            self.rf_model.train_ensemble_model(X_train, y_train, minority_threshold)
        else:
            # Train single RF model
            self.rf_model.train_model(X_train, y_train)
        
        train_time = time.time() - train_start
        print(f"✅ Training completed in {train_time:.1f}s")
        
        # Evaluate on training set
        print("\n📈 Evaluating on training set...")
        train_accuracy, _ = self.rf_model.evaluate_model(X_train, y_train)
        
        # Evaluate on test set
        print("📈 Evaluating on test set...")
        test_accuracy, y_pred = self.rf_model.evaluate_model(X_test, y_test)
        
        # Save model if requested (before cross validation)
        if save_model:
            print("\n💾 Saving model as requested...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_name = f"models/model_{timestamp}.pkl"
            scaler_name = f"models/scaler_{timestamp}.pkl"
            self.rf_model.save_model(model_name, scaler_name)
        
        # Cross validation if requested
        cv_results = None
        if config['use_cv']:
            # Prepare full dataset for CV
            df_processed = self.rf_model.apply_feature_engineering(self.data.copy()) if config['feature_engineering'] else self.data.copy()
            X_full = df_processed.drop(columns=['LABEL'])
            y_full = df_processed['LABEL']
            
            if config['manual_selection']:
                X_full = self.rf_model.select_specific_features(X_full)
            
            X_full_scaled = self.rf_model.scaler.fit_transform(X_full.values)
            cv_results = self.perform_cross_validation(X_full_scaled, y_full, config['cv_folds'])
        
        # Analyze results
        report = classification_report(y_test, y_pred, output_dict=True)
        
        total_time = time.time() - start_time
        
        # Compile results
        results = {
            'config': config,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_results': cv_results,
            'train_time': train_time,
            'total_time': total_time,
            'dataset_sizes': {
                'train': X_train.shape,
                'test': X_test.shape
            },
            'classification_report': report,
            'feature_count': X_train.shape[1],
            'model_type': model_type
        }
        
        # Display results
        self.display_results(results)
        
        # Store in history
        self.results_history.append(results)
        
        return results
    
    def display_results(self, results):
        """Display evaluation results"""
        model_type = results.get('model_type', 'single')
        
        print(f"\n🎉 EVALUATION RESULTS")
        print("="*50)
        
        if model_type == 'ensemble':
            print(f"🚀 Model Type: Advanced Stacking Ensemble")
            print(f"🧠 Base Models: XGBoost, LightGBM, CatBoost, RF, ExtraTrees")
            minority_threshold = results['config'].get('minority_threshold', 0.05)
            print(f"🎯 Minority Threshold: {minority_threshold:.2%}")
        else:
            print(f"🌳 Model Type: Single Random Forest")
            
        print(f"🏋️ Train Accuracy: {results['train_accuracy']:.4f}")
        print(f"🏆 Test Accuracy: {results['test_accuracy']:.4f}")
        
        # Check for overfitting based on train/test gap
        train_test_gap = results['train_accuracy'] - results['test_accuracy']
        if train_test_gap > 0.05:
            print(f"⚠️ Possible overfitting (train-test gap: {train_test_gap:.4f})")
        elif train_test_gap < -0.02:
            print(f"⚠️ Unusual: test > train (gap: {train_test_gap:.4f})")
        else:
            print(f"✅ Good train/test balance (gap: {train_test_gap:.4f})")
        
        if results['cv_results']:
            cv = results['cv_results']
            print(f"🔄 CV Accuracy: {cv['mean']:.4f} ± {cv['std']:.4f}")
            
            # Check CV vs test consistency
            cv_test_gap = abs(results['test_accuracy'] - cv['mean'])
            if cv_test_gap > 0.05:
                status = "⚠️ CV-test inconsistency"
            else:
                status = "✅ CV-test consistency"
            print(f"🔍 {status} (CV-test gap: {cv_test_gap:.4f})")
        
        print(f"🎯 Features used: {results['feature_count']}")
        print(f"⏱️ Training time: {results['train_time']:.1f}s")
        print(f"⏱️ Total time: {results['total_time']:.1f}s")
        
        # Note about confusion matrix visualizations
        print(f"\n📊 Visualizations Generated:")
        print(f"   🔍 Confusion matrices saved to visualizations/ folder")
        print(f"   📈 Feature importance plot generated")
        
        # Show class-wise performance for classes with poor performance
        print(f"\n📊 Performance Summary:")
        report = results['classification_report']
        poor_performers = []
        
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                f1 = metrics['f1-score']
                if f1 < 0.8:  # Flag classes with F1 < 0.8
                    poor_performers.append((class_name, f1))
        
        if poor_performers:
            print("⚠️ Classes needing attention (F1 < 0.8):")
            for class_name, f1 in sorted(poor_performers, key=lambda x: x[1]):
                print(f"  - {class_name}: F1 = {f1:.3f}")
        else:
            print("✅ All classes performing well (F1 ≥ 0.8)")
    
    def compare_configurations(self):
        """Run and compare multiple configurations"""
        print("\n📊 CONFIGURATION COMPARISON MODE")
        print("You'll configure multiple setups and compare their performance.")
        
        configurations = []
        config_names = []
        
        while True:
            print(f"\n--- Configuration #{len(configurations) + 1} ---")
            
            # Get configuration name
            name = input("Enter a name for this configuration: ").strip()
            if not name:
                name = f"Config_{len(configurations) + 1}"
            
            # Get all choices
            fe = self.get_feature_engineering_choice()
            fs = self.get_feature_selection_choice()
            balance_classes, balance_method = self.get_balancing_choice()
            use_cv, cv_folds = self.get_cross_validation_choice()
            
            config = {
                'feature_engineering': fe,
                'manual_selection': fs,
                'balance_classes': balance_classes,
                'balance_method': balance_method,
                'use_cv': use_cv,
                'cv_folds': cv_folds
            }
            
            configurations.append(config)
            config_names.append(name)
            
            # Ask if they want to add another
            while True:
                another = input("\nAdd another configuration? (y/n): ").strip().lower()
                if another in ['y', 'yes']:
                    break
                elif another in ['n', 'no']:
                    break
                else:
                    print("❌ Please enter 'y' or 'n'")
            
            if another in ['n', 'no']:
                break
        
        # Run all configurations
        print(f"\n🚀 Running {len(configurations)} configurations...")
        all_results = []
        
        for i, (config, name) in enumerate(zip(configurations, config_names)):
            separator = "=" * 20
            print(f"\n{separator} {name} ({i+1}/{len(configurations)}) {separator}")
            results = self.run_evaluation(config)
            if results:
                results['name'] = name
                all_results.append(results)
        
        # Compare results
        if len(all_results) > 1:
            self.display_comparison(all_results)
    
    def save_best_model(self, best_result):
        """Save the best model with descriptive naming"""
        try:
            # Create descriptive filename based on configuration
            config = best_result['config']
            
            # Build filename components
            fe_suffix = "_FE" if config['feature_engineering'] else "_noFE"
            fs_suffix = "_Manual" if config['manual_selection'] else "_AllFeatures"
            balance_suffix = f"_{config['balance_method']}" if config['balance_classes'] else "_noBalance"
            cv_suffix = f"_CV{config['cv_folds']}" if config['use_cv'] else "_noCV"
            accuracy_suffix = f"_acc{best_result['test_accuracy']:.4f}"
            
            # Create model filename
            model_filename = f"models/best_model{fe_suffix}{fs_suffix}{balance_suffix}{cv_suffix}{accuracy_suffix}.pkl"
            scaler_filename = f"models/best_scaler{fe_suffix}{fs_suffix}{balance_suffix}{cv_suffix}{accuracy_suffix}.pkl"
            
            # Save the model
            print(f"\n💾 Saving best model...")
            self.rf_model.save_model(model_filename, scaler_filename)
            
            # Save configuration details
            config_filename = f"models/best_config{fe_suffix}{fs_suffix}{balance_suffix}{cv_suffix}{accuracy_suffix}.txt"
            with open(config_filename, 'w') as f:
                f.write(f"Best Model Configuration\n")
                f.write(f"========================\n")
                f.write(f"Configuration Name: {best_result['name']}\n")
                f.write(f"Train Accuracy: {best_result['train_accuracy']:.4f}\n")
                f.write(f"Test Accuracy: {best_result['test_accuracy']:.4f}\n")
                if best_result['cv_results']:
                    f.write(f"CV Accuracy: {best_result['cv_results']['mean']:.4f} ± {best_result['cv_results']['std']:.4f}\n")
                f.write(f"Feature Engineering: {'Yes' if config['feature_engineering'] else 'No'}\n")
                f.write(f"Feature Selection: {'Manual (16 features)' if config['manual_selection'] else 'All features'}\n")
                f.write(f"Class Balancing: {config['balance_method'] if config['balance_classes'] else 'None'}\n")
                cv_config_text = f"{config['cv_folds']}-fold" if config['use_cv'] else 'None'
                f.write(f"Cross Validation: {cv_config_text}\n")
                
                # Add ensemble information
                model_type = best_result.get('model_type', 'single')
                if model_type == 'ensemble':
                    f.write(f"Model Type: Advanced Stacking Ensemble\n")
                    f.write(f"Base Models: XGBoost, LightGBM, CatBoost, RandomForest, ExtraTrees\n")
                    f.write(f"Meta-Learner: XGBoost\n")
                    minority_threshold = config.get('minority_threshold', 0.05)
                    f.write(f"Minority Threshold: {minority_threshold:.2%}\n")
                else:
                    f.write(f"Model Type: Single Random Forest\n")
                
                f.write(f"Features Used: {best_result['feature_count']}\n")
                f.write(f"Training Time: {best_result['train_time']:.1f}s\n")
                f.write(f"Total Time: {best_result['total_time']:.1f}s\n")
            
            print(f"✅ Best model saved as: {model_filename}")
            print(f"✅ Best scaler saved as: {scaler_filename}")
            print(f"✅ Configuration saved as: {config_filename}")
            
        except Exception as e:
            print(f"❌ Error saving best model: {e}")

    def display_comparison(self, results_list):
        """Display comparison of multiple results and save the best model"""
        print(f"\n📊 CONFIGURATION COMPARISON")
        print("="*80)
        
        # Sort by test accuracy
        results_list.sort(key=lambda x: x['test_accuracy'], reverse=True)
        
        print(f"{'Rank':<4} {'Name':<20} {'Train Acc':<10} {'Test Acc':<10} {'CV Acc':<12} {'Features':<8} {'Type':<8} {'Time(s)':<8}")
        print("-" * 98)
        
        for i, result in enumerate(results_list):
            cv_str = f"{result['cv_results']['mean']:.4f}" if result['cv_results'] else "N/A"
            model_type = result.get('model_type', 'single')
            type_str = "Stacking" if model_type == 'ensemble' else "RF"
            print(f"{i+1:<4} {result['name']:<20} {result['train_accuracy']:<10.4f} {result['test_accuracy']:<10.4f} {cv_str:<12} "
                  f"{result['feature_count']:<8} {type_str:<8} {result['train_time']:<8.1f}")
        
        # Show best configuration details
        best = results_list[0]
        best_model_type = best.get('model_type', 'single')
        print(f"\n🏆 BEST CONFIGURATION: {best['name']}")
        print(f"  🔧 Feature Engineering: {'✅' if best['config']['feature_engineering'] else '❌'}")
        print(f"  🎯 Feature Selection: {'Manual' if best['config']['manual_selection'] else 'All'}")
        print(f"  ⚖️ Balancing: {best['config']['balance_method'] if best['config']['balance_classes'] else 'None'}")
        best_cv_text = f"{best['config']['cv_folds']}-fold" if best['config']['use_cv'] else 'None'
        print(f"  🔄 Cross Validation: {best_cv_text}")
        
        if best_model_type == 'ensemble':
            print(f"  🚀 Model Type: Advanced Stacking Ensemble")
            print(f"  🧠 Base Models: XGBoost, LightGBM, CatBoost, RF, ExtraTrees")
            minority_threshold = best['config'].get('minority_threshold', 0.05)
            print(f"  🎯 Minority Threshold: {minority_threshold:.2%}")
        else:
            print(f"  🌳 Model Type: Single Random Forest")
            
        print(f"  🏋️ Train Accuracy: {best['train_accuracy']:.4f}")
        print(f"  🏆 Test Accuracy: {best['test_accuracy']:.4f}")
        
        if best['cv_results']:
            print(f"  🔄 CV Accuracy: {best['cv_results']['mean']:.4f} ± {best['cv_results']['std']:.4f}")
        
        # Ask user if they want to save the best model
        while True:
            save_choice = input(f"\n💾 Save the best model ('{best['name']}')? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                # Re-train the best model to ensure it's saved
                print(f"\n🔄 Re-training best configuration for saving...")
                best_results = self.run_evaluation(best['config'], save_model=True)
                if best_results:
                    self.save_best_model(best_results)
                break
            elif save_choice in ['n', 'no']:
                print("📝 Best model not saved.")
                break
            else:
                print("❌ Please enter 'y' or 'n'")
    
    def view_results_history(self):
        """View previous evaluation results"""
        if not self.results_history:
            print("\n📝 No evaluation results in history yet.")
            return
        
        print(f"\n📈 RESULTS HISTORY ({len(self.results_history)} evaluations)")
        print("="*60)
        
        for i, result in enumerate(self.results_history):
            model_type = result.get('model_type', 'single')
            type_str = "Stacking" if model_type == 'ensemble' else "RF"
            print(f"\n{i+1}. Evaluation #{i+1} ({type_str})")
            print(f"   🏋️ Train Accuracy: {result['train_accuracy']:.4f}")
            print(f"   🏆 Test Accuracy: {result['test_accuracy']:.4f}")
            if result['cv_results']:
                print(f"   🔄 CV Accuracy: {result['cv_results']['mean']:.4f}")
            print(f"   🎯 Features: {result['feature_count']}")
            if model_type == 'ensemble':
                minority_threshold = result['config'].get('minority_threshold', 0.05)
                print(f"   🚀 Minority Threshold: {minority_threshold:.2%}")
            print(f"   ⏱️ Time: {result['train_time']:.1f}s")
    
    def analyze_class_distribution(self):
        """Analyze and display class distribution"""
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n📊 CLASS DISTRIBUTION ANALYSIS")
        print("="*50)
        
        class_counts = self.data['LABEL'].value_counts().sort_values()
        total_samples = len(self.data)
        
        print(f"Total samples: {total_samples}")
        print(f"Number of classes: {len(class_counts)}")
        print("\nClass distribution (least to most represented):")
        
        for class_id, count in class_counts.items():
            class_name = self.rf_model.label_mapping[class_id]
            percentage = (count / total_samples) * 100
            bar = "█" * int(percentage / 2)  # Visual bar
            print(f"{class_name:<25}: {count:>6} ({percentage:>5.2f}%) {bar}")
        
        # Identify problematic classes
        minority_threshold = total_samples * 0.02  # Less than 2%
        majority_threshold = total_samples * 0.10  # More than 10%
        
        minority_classes = [self.rf_model.label_mapping[cid] for cid, count in class_counts.items() if count < minority_threshold]
        majority_classes = [self.rf_model.label_mapping[cid] for cid, count in class_counts.items() if count > majority_threshold]
        
        if minority_classes:
            print(f"\n⚠️ Minority classes (<2%): {', '.join(minority_classes)}")
        
        if majority_classes:
            print(f"\n📈 Majority classes (>10%): {', '.join(majority_classes)}")
    
    def run_ensemble_evaluation(self):
        """Run advanced stacking ensemble model evaluation"""
        print("\n🚀 ADVANCED STACKING ENSEMBLE CONFIGURATION")
        print("🎯 XGBoost + LightGBM + CatBoost + RandomForest + ExtraTrees")
        print("🧠 With XGBoost Meta-Learner for Final Predictions")
        print("="*70)
        
        # Get configuration choices
        fe = self.get_feature_engineering_choice()
        fs = self.get_feature_selection_choice()
        balance_classes, balance_method = self.get_balancing_choice()
        use_cv, cv_folds = self.get_cross_validation_choice()
        
        # Get minority class threshold
        print("\n🎯 MINORITY CLASS THRESHOLD:")
        print("Classes below this threshold will benefit from ensemble optimization")
        while True:
            try:
                threshold = float(input("Enter minority threshold (0.01-0.10, default 0.05): ").strip() or "0.05")
                if 0.01 <= threshold <= 0.10:
                    break
                else:
                    print("❌ Please enter a value between 0.01 and 0.10")
            except ValueError:
                print("❌ Please enter a valid number")
        
        config = {
            'feature_engineering': fe,
            'manual_selection': fs,
            'balance_classes': balance_classes,
            'balance_method': balance_method,
            'use_cv': use_cv,
            'cv_folds': cv_folds,
            'minority_threshold': threshold,
            'model_type': 'ensemble'
        }
        
        # Ask if user wants to save the model
        while True:
            save_ensemble = input("\n💾 Save this advanced stacking ensemble? (y/n): ").strip().lower()
            if save_ensemble in ['y', 'yes']:
                self.run_evaluation(config, save_model=True)
                break
            elif save_ensemble in ['n', 'no']:
                self.run_evaluation(config, save_model=False)
                break
            else:
                print("❌ Please enter 'y' or 'n'")
    
    def run(self):
        """Main execution loop"""
        if not self.load_data():
            return
        
        while True:
            self.display_main_menu()
            
            try:
                choice = input("\nSelect option (1-6): ").strip()
                
                if choice == "1":
                    # Single evaluation
                    print("\n🔧 SINGLE EVALUATION CONFIGURATION")
                    
                    fe = self.get_feature_engineering_choice()
                    fs = self.get_feature_selection_choice()
                    balance_classes, balance_method = self.get_balancing_choice()
                    use_cv, cv_folds = self.get_cross_validation_choice()
                    
                    config = {
                        'feature_engineering': fe,
                        'manual_selection': fs,
                        'balance_classes': balance_classes,
                        'balance_method': balance_method,
                        'use_cv': use_cv,
                        'cv_folds': cv_folds
                    }
                    
                    # Ask if user wants to save the model
                    while True:
                        save_single = input("\n💾 Save this model? (y/n): ").strip().lower()
                        if save_single in ['y', 'yes']:
                            self.run_evaluation(config, save_model=True)
                            break
                        elif save_single in ['n', 'no']:
                            self.run_evaluation(config, save_model=False)
                            break
                        else:
                            print("❌ Please enter 'y' or 'n'")
                
                elif choice == "2":
                    # Compare configurations
                    self.compare_configurations()
                
                elif choice == "3":
                    # View history
                    self.view_results_history()
                
                elif choice == "4":
                    # Analyze class distribution
                    self.analyze_class_distribution()
                
                elif choice == "5":
                    # Ensemble model evaluation
                    self.run_ensemble_evaluation()
                
                elif choice == "6":
                    # Exit
                    print("\n👋 Thank you for using the Model Evaluator!")
                    break
                
                else:
                    print("❌ Invalid choice. Please select 1-6.")
                
                # Pause before showing menu again
                if choice in ["1", "2", "3", "4", "5"]:
                    input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("Press Enter to continue...")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run() 