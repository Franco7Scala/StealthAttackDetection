import torch
import random
import numpy as np
import os
import pandas as pd
import json

def get_base_dir():
    return "/home/jovyan/StealthAttackDetection"


def set_reproducibility(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False


def compute_scale_pos_weight(y_train_np):
    n_pos = (y_train_np == 1).sum()
    n_neg = (y_train_np == 0).sum()
    return n_neg / n_pos



def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True

    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False

    else:
        raise Exception("Boolean value expected.")



def print_args(args):
    args_dict = vars(args)
    if not args_dict:
        print("No arguments provided.")
        return

    max_key_len = max(len(key) for key in args_dict)
    print("\n" + "=" * 30)
    print(f"{'CONFIGURATION':^{30}}")
    print("=" * 30)
    for key, value in sorted(args_dict.items()):
        print(f"{key:<{max_key_len}} : {value}")

    print("=" * 30 + "\n")




def aggregate_runs(base_dir, attack_type, keyword=None):

    attack_path = os.path.join(base_dir, attack_type)

    if keyword is None:

        all_files = [
            os.path.join(attack_path, f)
            for f in os.listdir(attack_path)
            if f.endswith(".csv")
        ]

    else:

        all_files = [
            os.path.join(attack_path, f)
            for f in os.listdir(attack_path)
            if f.startswith("run_")
            and keyword in f
            and f.endswith(".csv")
        ]

    if not all_files:
        raise ValueError(f"No run files found in {attack_path}")

    df_list = [pd.read_csv(f) for f in all_files]

    df = pd.concat(df_list, ignore_index=True)

    exclude_cols = ["run_id", "attack_type", "model"]

    metric_cols = [
        c for c in df.columns
        if c not in exclude_cols
    ]

    grouped = df.groupby("model")[metric_cols]

    mean_df = grouped.mean().add_suffix("_mean")
    std_df = grouped.std().add_suffix("_std")

    result = pd.concat([mean_df, std_df], axis=1).reset_index()

    for col in metric_cols:

        result[f"{col}_mean±std"] = (
            result[f"{col}_mean"].round(4).astype(str)
            + " ± "
            + result[f"{col}_std"].round(4).astype(str)
        )

    return result







import os
import json
import numpy as np
import re

def compute_pr_auc_stats(
    root_dir,
    experiment_name,
    prc_folder="prc_model",
    json_name="prc_metrics_train.json",
    pr_auc_key="pr_auc"
):

    pr_auc_values = []

    prc_path = os.path.join(root_dir, prc_folder)

    if not os.path.exists(prc_path):
        raise ValueError(f"Directory non trovata: {prc_path}")

    pattern = re.compile(re.escape(experiment_name))  # match flessibile

    for run_dir in os.listdir(prc_path):

        run_path = os.path.join(prc_path, run_dir)

        if not os.path.isdir(run_path):
            continue

    
        if not pattern.search(run_dir):
            continue

        json_path = os.path.join(run_path, json_name)

        if not os.path.exists(json_path):
            print(f"[WARNING] JSON mancante in {run_dir}")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            value = data.get(pr_auc_key, None)

            if isinstance(value, (int, float)):
                pr_auc_values.append(value)
                print(f"[OK] {run_dir} -> pr_auc = {value}")
            else:
                print(f"[WARNING] valore non valido in {run_dir}")

        except json.JSONDecodeError:
            print(f"[ERROR] JSON corrotto: {json_path}")

    if len(pr_auc_values) == 0:
        raise ValueError("Nessun valore pr_auc trovato.")

    mean_value = np.mean(pr_auc_values)
    std_value = np.std(pr_auc_values)

    return {
        "num_runs": len(pr_auc_values),
        "values": pr_auc_values,
        "mean": mean_value,
        "std": std_value,
        "mean_plus_std": f"{mean_value} ± {std_value}"
    }