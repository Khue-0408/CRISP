"""
Teacher model wrappers and ensemble execution helpers.

Teachers are used only during training to form the boundary-local posterior target.
At inference, teachers are not used. [file:1]

CRISP reference: instruct.md §4, §13, §14.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from crisp.utils.model_loading import load_model_checkpoint


class FrozenTeacher(nn.Module):
    """
    Thin wrapper around a pretrained segmentation model used as a frozen teacher.

    Responsibilities
    ----------------
    - load teacher checkpoint,
    - enforce evaluation mode,
    - expose probability maps rather than raw logits if desired,
    - ensure gradients are never computed through teacher parameters.

    CRISP reference
    ---------------
    instruct.md §4.1: Teachers are frozen during CRISP training.
    instruct.md §13: Teacher outputs must be detached.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: str,
        checkpoint_loading: Dict | None = None,
        auto_download: bool = False,
        download_url: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.checkpoint_loading = checkpoint_loading or {}
        self.auto_download = auto_download
        self.download_url = download_url
        self._load_and_freeze()

    def _load_and_freeze(self) -> None:
        """Load checkpoint, freeze all parameters, and set eval mode."""
        if self.checkpoint_path:
            load_model_checkpoint(
                self.model,
                self.checkpoint_path,
                strict=bool(self.checkpoint_loading.get("strict", True)),
                state_dict_keys=self.checkpoint_loading.get("state_dict_keys"),
                prefixes_to_strip=self.checkpoint_loading.get("prefixes_to_strip"),
                auto_download=self.auto_download,
                download_url=self.download_url,
                description=f"teacher checkpoint for {type(self.model).__name__}",
            )
        # Freeze all parameters.
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def train(self, mode: bool = True) -> "FrozenTeacher":
        """Override train to always stay in eval mode."""
        # Teachers are always frozen — never switch to training mode.
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the frozen teacher and return a foreground probability map.

        Parameters
        ----------
        x:
            Input batch of images [B, 3, H, W].

        Returns
        -------
        torch.Tensor
            Foreground probability tensor of shape [B, 1, H, W], detached.
        """
        out = self.model(x)
        # Support both SegmentationOutput dataclass and raw tensor returns.
        if hasattr(out, "logits"):
            logits = out.logits
        elif isinstance(out, torch.Tensor):
            logits = out
        else:
            raise ValueError(f"Unexpected teacher output type: {type(out)}")

        probs = torch.sigmoid(logits)
        return probs.detach()


class TeacherEnsemble(nn.Module):
    """
    Container for multiple frozen teacher models.

    This class centralizes all teacher execution so the training loop can request
    a list or stacked tensor of teacher probability maps in one call.
    """

    def __init__(self, teachers: List[FrozenTeacher]) -> None:
        super().__init__()
        self.teachers = nn.ModuleList(teachers)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Run all teachers on the input batch.

        Returns
        -------
        List[torch.Tensor]
            List of probability maps [B, 1, H, W], one per teacher.
            All outputs are detached.
        """
        return [teacher(x) for teacher in self.teachers]
