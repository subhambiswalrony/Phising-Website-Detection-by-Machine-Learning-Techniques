"""
Retrain XGBoost Model - Fix Version Compatibility
This script retrains the XGBoost model with the current XGBoost version (3.2.0)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

def retrain_model():
    print("=" * 70)
    print(" RETRAINING XGBOOST MODEL FOR VERSION COMPATIBILITY")
    print("=" * 70)
    print()
    
    # Step 1: Load dataset
    print("Step 1: Loading dataset...")
    try:
        data = pd.read_csv('DataFiles/urldata.csv')
        print(f"✓ Dataset loaded: {data.shape[0]} rows × {data.shape[1]} columns")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False
    
    # Step 2: Prepare data
    print("\nStep 2: Preparing data...")
    try:
        # Drop Domain column (not needed for training)
        if 'Domain' in data.columns:
            X = data.drop(['Domain', 'Label'], axis=1)
        else:
            X = data.drop(['Label'], axis=1)
        
        y = data['Label']
        
        print(f"✓ Features: {X.shape[1]}")
        print(f"✓ Samples: {X.shape[0]}")
        print(f"✓ Phishing: {(y == 1).sum()}, Legitimate: {(y == 0).sum()}")
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        return False
    
    # Step 3: Split data (same as original: 80/20)
    print("\nStep 3: Splitting data (80% train, 20% test)...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=12
        )
        print(f"✓ Train set: {X_train.shape[0]} samples")
        print(f"✓ Test set: {X_test.shape[0]} samples")
    except Exception as e:
        print(f"✗ Error splitting data: {e}")
        return False
    
    # Step 4: Train XGBoost model (same parameters as original)
    print("\nStep 4: Training XGBoost model...")
    print("   Parameters: learning_rate=0.4, max_depth=7")
    try:
        xgb_model = XGBClassifier(learning_rate=0.4, max_depth=7)
        xgb_model.fit(X_train, y_train)
        print("✓ Model trained successfully")
    except Exception as e:
        print(f"✗ Error training model: {e}")
        return False
    
    # Step 5: Evaluate model
    print("\nStep 5: Evaluating model performance...")
    try:
        # Training accuracy
        y_train_pred = xgb_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Testing accuracy
        y_test_pred = xgb_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"✓ Training Accuracy: {train_accuracy:.3f} ({train_accuracy*100:.1f}%)")
        print(f"✓ Testing Accuracy:  {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
        
        # Detailed metrics
        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(y_test, y_test_pred, 
                                  target_names=['Legitimate', 'Phishing']))
        
        print("\n" + "=" * 70)
        print("CONFUSION MATRIX")
        print("=" * 70)
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"                 Predicted")
        print(f"                 Legitimate  Phishing")
        print(f"Actual Legitimate  {cm[0][0]:>6}     {cm[0][1]:>6}")
        print(f"       Phishing    {cm[1][0]:>6}     {cm[1][1]:>6}")
        
    except Exception as e:
        print(f"✗ Error evaluating model: {e}")
        return False
    
    # Step 6: Save model
    print("\n" + "=" * 70)
    print("Step 6: Saving model...")
    try:
        # Backup old model if exists
        if os.path.exists('XGBoostClassifier.pickle.dat'):
            # Remove old backup if it exists
            if os.path.exists('XGBoostClassifier.pickle.dat.backup'):
                os.remove('XGBoostClassifier.pickle.dat.backup')
            os.rename('XGBoostClassifier.pickle.dat', 
                     'XGBoostClassifier.pickle.dat.backup')
            print("✓ Old model backed up as 'XGBoostClassifier.pickle.dat.backup'")
        
        # Save new model
        with open('XGBoostClassifier.pickle.dat', 'wb') as f:
            pickle.dump(xgb_model, f)
        
        file_size = os.path.getsize('XGBoostClassifier.pickle.dat') / 1024
        print(f"✓ New model saved: XGBoostClassifier.pickle.dat ({file_size:.2f} KB)")
        
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return False
    
    # Step 7: Verify model loading
    print("\nStep 7: Verifying model can be loaded...")
    try:
        with open('XGBoostClassifier.pickle.dat', 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Quick test
        test_pred = loaded_model.predict(X_test[:5])
        print(f"✓ Model loaded successfully")
        print(f"✓ Test prediction on 5 samples: {test_pred}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    # Success!
    print("\n" + "=" * 70)
    print("✅ MODEL RETRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel Performance:")
    print(f"  Training Accuracy: {train_accuracy*100:.1f}%")
    print(f"  Testing Accuracy:  {test_accuracy*100:.1f}%")
    print(f"\nModel saved to: XGBoostClassifier.pickle.dat")
    print(f"\nYou can now use the model for predictions!")
    print("\nExample usage:")
    print("  import pickle")
    print("  with open('XGBoostClassifier.pickle.dat', 'rb') as f:")
    print("      model = pickle.load(f)")
    print("  predictions = model.predict(X_test)")
    print()
    
    return True


if __name__ == "__main__":
    success = retrain_model()
    
    if not success:
        print("\n" + "=" * 70)
        print("❌ RETRAINING FAILED")
        print("=" * 70)
        print("\nPlease check the errors above and try again.")
        print("Make sure DataFiles/urldata.csv exists and is readable.")
    
    input("\nPress Enter to exit...")
