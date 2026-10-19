"""SPE MLP training and evaluation."""
from pathlib import Path


if __name__ == "__main__":
    from User_data_processing import SingleUserEvaluation

    Path("mlp").mkdir(parents=True, exist_ok=True)
    SingleUserEvaluation("./preprocessed", "mlp", 100, 50, 0.01)
