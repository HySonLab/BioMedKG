import numpy as np
import pandas as pd
import xgboost as xgb
from lightning.pytorch import seed_everything
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from biomedkg.data.node import KGEEncode

seed_everything(42)


def main(ckpt_path: str, node_init_method: str, gcl_model: str, gcl_fuse_method: str):
    kge_encode = KGEEncode(
        ckpt_path=ckpt_path,
        node_init_method=node_init_method,
        gcl_model=gcl_model,
        gcl_fuse_method=gcl_fuse_method,
    )

    data_dir = "data/dpi/dpi_benchmark.csv"

    df = pd.read_csv(
        data_dir,
    )

    df = df.dropna()

    node_names = set(df["x_name"].values) | set(df["y_name"].values)
    node_names = list(node_names)
    node_embeddings = kge_encode(node_names)

    num_nodes = len(df)

    dpi_node_mapping = dict()

    for idx in range(len(node_names)):
        dpi_node_mapping[node_names[idx]] = (
            node_embeddings[idx].squeeze(0).cpu().numpy()
        )

    head_embeddings = np.array(
        [dpi_node_mapping[node] for node in df["x_name"].values.tolist()]
    )
    tail_embeddings = np.array(
        [dpi_node_mapping[node] for node in df["y_name"].values.tolist()]
    )
    pos_embeddings = np.concatenate(
        [
            np.expand_dims(head_embeddings, axis=1),
            np.expand_dims(tail_embeddings, axis=1),
        ],
        axis=1,
    )
    pos_labels = np.ones((pos_embeddings.shape[0]))

    neg_head_embeddings_indices = np.random.choice(
        num_nodes, size=3 * num_nodes, replace=True
    )
    neg_tail_embeddings_indices = np.random.choice(
        num_nodes, size=3 * num_nodes, replace=True
    )

    neg_head_embeddings = head_embeddings[neg_head_embeddings_indices]
    neg_tail_embeddings = tail_embeddings[neg_tail_embeddings_indices]

    neg_embeddings = np.concatenate(
        [
            np.expand_dims(neg_head_embeddings, axis=1),
            np.expand_dims(neg_tail_embeddings, axis=1),
        ],
        axis=1,
    )
    neg_labels = np.zeros((neg_embeddings.shape[0]))

    X = np.concatenate([pos_embeddings, neg_embeddings], axis=0)
    y = np.concatenate([pos_labels, neg_labels], axis=0)

    X_mean = np.mean(X, axis=1)

    # Initialize Stratified K-Fold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize lists to store metrics
    f1_scores = []
    average_precisions = []

    # Perform Stratified K-Fold Cross-Validation
    for train_index, val_index in skf.split(X_mean, y):
        X_train, X_val = X_mean[train_index], X_mean[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = xgb.XGBClassifier(
            device="cuda",
            n_estimators=500,
            max_depth=5,
            learning_rate=0.01,
            random_state=42,
        )

        # Train the model
        model.fit(X_train, y_train)

        # Predict on the validation set
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[
            :, 1
        ]  # Probabilities for Average Precision

        # Compute F1-score
        f1 = f1_score(y_val, y_val_pred, pos_label=1)
        f1_scores.append(f1)

        # Compute Average Precision
        ap = average_precision_score(y_val, y_val_proba)
        average_precisions.append(ap)

    # Compute mean metrics
    mean_f1 = sum(f1_scores) / n_splits
    mean_ap = sum(average_precisions) / n_splits

    # Print results
    print(f"Result for {ckpt_path}")
    print(f"F1-Scores for each fold: {f1_scores}")
    print(f"Average Precision for each fold: {average_precisions}")
    print(f"Mean F1-Score: {mean_f1:.4f}")
    print(f"Mean Average Precision (AP): {mean_ap:.4f}")
    print("=" * 20)


if __name__ == "__main__":
    configs = [
        {
            "ckpt_path": "ckpt/path/to/best.ckpt",
            "node_init_method": "random",
            "gcl_model": "grace",
            "gcl_fuse_method": "none",
        },
        {
            "ckpt_path": "ckpt/path/to/best.ckpt",
            "node_init_method": "lm",
            "gcl_model": "grace",
            "gcl_fuse_method": "none",
        },
        {
            "ckpt_path": "ckpt/path/to/best.ckpt",
            "node_init_method": "gcl",
            "gcl_model": "grace",
            "gcl_fuse_method": "attention",
        },
    ]

    for config in tqdm(configs):
        main(**config)
