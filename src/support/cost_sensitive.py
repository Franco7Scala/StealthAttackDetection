# custom_loss.py

import torch
import torch.nn as nn


class CostSensitiveLoss(nn.Module):
    """
    Cost Sensitive Loss.

    Penalizza diversamente:
    - False Negatives (fn_weight)
    - False Positives (fp_weight)

    Se prediction == target viene applicato peso 1.

    Args:
        fn_weight (float): peso dei falsi negativi.
        fp_weight (float): peso dei falsi positivi.
    """

    def __init__(self, fn_weight=1.0, fp_weight=1.0):
        super().__init__()
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (Tensor): predizioni del modello.
            y_true (Tensor): target.

        Returns:
            Tensor: loss scalare.
        """

        # False Negatives: y_true > y_pred
        mask_fn = torch.clamp(torch.round(y_true - y_pred), min=0, max=1)
        w_fn = mask_fn * self.fn_weight

        # False Positives: y_pred > y_true
        mask_fp = torch.clamp(torch.round(y_pred - y_true), min=0, max=1)
        w_fp = mask_fp * self.fp_weight

        # Predizioni corrette
        mask_other = torch.clamp(
            1 - torch.round(torch.abs(y_true - y_pred)),
            min=0,
            max=1,
        )

        weights = w_fn + w_fp + mask_other

        loss = torch.mean((y_pred - y_true) ** 2 * weights)

        return loss