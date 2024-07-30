import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection, Metric
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class EdgeWisePrecision(Metric):
    def __init__(self, class_mapping: dict, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.class_mapping = class_mapping
        self.num_classes = len(class_mapping)
        self.add_state("class_counts", default=torch.zeros(self.num_classes), dist_reduce_fx="sum")
        self.add_state("above_threshold_counts", default=torch.zeros(self.num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        for class_idx in range(self.num_classes):
            class_mask = (target == class_idx)
            self.class_counts[class_idx] += class_mask.sum()
            self.above_threshold_counts[class_idx] += (preds[class_mask] > self.threshold).sum()

    def compute(self) -> dict:
        percentages = {}
        for class_idx in range(self.num_classes):
            key_name = self.class_mapping[class_idx] + "_pred"
            if self.class_counts[class_idx] > 0:
                percentages[key_name] = (self.above_threshold_counts[class_idx] / self.class_counts[class_idx]).item()
            else:
                percentages[key_name] = 0.0
        return percentages

class HitAtK(Metric):
    def __init__(self, k: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.add_state("hits", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # Ensure preds and target are 2D tensors
        if preds.dim() == 1:
            preds = preds.unsqueeze(0)
            target = target.unsqueeze(0)
        
        for p, t in zip(preds, target):
            _, topk_indices = torch.topk(p, self.k, largest=True, sorted=True)
            self.hits += (t[topk_indices] == 1).any().float()
            self.total += 1.0

    def compute(self) -> float:
        return (self.hits / self.total).item()
    
class MeanReciprocalRank(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("reciprocal_ranks", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # Ensure preds and target are 2D tensors
        if preds.dim() == 1:
            preds = preds.unsqueeze(0)
            target = target.unsqueeze(0)

        for p, t in zip(preds, target):
            sorted_indices = torch.argsort(p, descending=True)
            relevant_indices = (t == 1).nonzero(as_tuple=True)[0]
            if len(relevant_indices) > 0:
                ranks = torch.where(sorted_indices.unsqueeze(1) == relevant_indices)[0] + 1
                rank = ranks.min().item()
                self.reciprocal_ranks += 1.0 / rank
            self.total += relevant_indices.numel()

    def compute(self) -> float:
        return (self.reciprocal_ranks / self.total).item()