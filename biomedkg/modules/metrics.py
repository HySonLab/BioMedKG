import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.functional.retrieval.hit_rate import retrieval_hit_rate
from torchmetrics.functional.retrieval.reciprocal_rank import retrieval_reciprocal_rank

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


class HitsAtK(Metric):
    """
    Computes the Hits@k metric for evaluating ranking or recommendation systems.
    Extends directly from torchmetrics.Metric for full customization.
    """

    def __init__(self, k: int = 10, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.k = k
        self.add_state("hits", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor): 

        """
        Updates the metric state with predictions and targets.
        """
        hits = retrieval_hit_rate(preds, target, self.k) 
        self.hits += hits.sum()
        self.total += target.size(0)

    def compute(self):
        """
        Computes the Hits@k metric over all accumulated predictions and targets.
        """
        return self.hits.float() / self.total
    

class MeanReciprocalRank(Metric):
    """
    Computes the Mean Reciprocal Rank (MRR) for evaluating ranking or recommendation systems.
    Extends directly from torchmetrics.Metric for full customization.
    """

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("rr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor): 

        """
        Updates the metric state with predictions and targets.
        """
        rr = retrieval_reciprocal_rank(preds, target)
        self.rr_sum += rr.sum()
        self.total += target.size(0)

    def compute(self):
        """
        Computes the MRR over all accumulated predictions and targets.
        """
        return self.rr_sum / self.total