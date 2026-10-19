"""Grad-CAM for the visual encoder.

Produces a spatial heatmap over the mouth ROI showing which pixels most
influenced the clip-level fake decision. Used as a complementary signal to
the temporal sync timeline in the demo and the paper figures.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, _input, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def __call__(self, video: torch.Tensor, audio: torch.Tensor, class_idx: int = 1) -> torch.Tensor:
        """Returns a (B, T, H, W) heatmap in [0, 1]."""
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)
        out = self.model(video, audio)
        score = out.logits[:, class_idx].sum()
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # activations: (B*T, C, H, W); gradients same shape
        act = self._activations
        grad = self._gradients
        if act is None or grad is None:
            raise RuntimeError("Grad-CAM hooks did not fire.")
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1)  # (B*T, H, W)
        cam = F.relu(cam)
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)

        # cam is (B*T, h_small, w_small). Upsample each frame's CAM to the
        # original mouth-ROI resolution (H, W), then reshape to (B, T, H, W).
        b = video.shape[0]; t = video.shape[1]
        cam = cam.unsqueeze(1)  # (B*T, 1, h_small, w_small)
        cam = F.interpolate(cam, size=video.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)    # (B*T, H, W)
        return cam.view(b, t, cam.shape[-2], cam.shape[-1])
