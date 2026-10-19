"""SUE SNN training and noise evaluation."""
from pathlib import Path


if __name__ == "__main__":
    from User_data_processing_attack import SingleUserEvaluation

    Path("snn").mkdir(parents=True, exist_ok=True)
    Path("model/mlp_3l").mkdir(parents=True, exist_ok=True)
    spike_log = Path("total_mean_values.txt")
    spike_log.write_text("")
    SingleUserEvaluation("../data/Study1_medfilt11_EMG", "mlp", 100, 50, 0.01,
                         loadpath=None, savepath="model/mlp_3l")
    values = [float(line) for line in spike_log.read_text().splitlines() if line.strip()]
    if values:
        print("Legacy batch-averaged spike rate:", sum(values) / len(values))
