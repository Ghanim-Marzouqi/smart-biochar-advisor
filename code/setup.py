"""
Setup script for first-time deployment
Trains the model if it doesn't exist
"""

import os
from pathlib import Path
import sys

def setup_model():
    """Train model if it doesn't exist."""
    model_path = Path(__file__).parent / "models" / "trained" / "pgi_model.pkl"

    if not model_path.exists():
        print("🔧 Model not found. Training model for first-time deployment...")

        # Import and run training
        sys.path.append(str(Path(__file__).parent))
        from models.train_model import train_all_models

        try:
            train_all_models()
            print("✅ Model training complete!")
            return True
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False
    else:
        print("✅ Model already exists")
        return True

if __name__ == "__main__":
    success = setup_model()
    sys.exit(0 if success else 1)
