import numpy as np
import torch
import torch.nn as nn

from drxai.xai.attribution import GradCAM, integrated_gradients
from drxai.xai.evaluation import cell_order, curve_auc, localisation_scores, map_similarity, perturbation_curve


class TinyNet(nn.Module):
    """conv -> ReLU -> global average pool -> linear: a minimal CAM-compatible model."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.features(x).mean((2, 3)))


def test_integrated_gradients_completeness():
    torch.manual_seed(0)
    model = TinyNet().eval()
    x, base = torch.rand(1, 3, 16, 16), torch.zeros(1, 3, 16, 16)
    attr, gap = integrated_gradients(model, x, base, target=1, steps=256, chunk=64)
    with torch.no_grad():
        delta = (model(x)[0, 1] - model(base)[0, 1]).item()
    assert attr.shape == (3, 16, 16)
    assert abs(attr.sum().item() - delta) < 1e-2 * max(1.0, abs(delta))
    assert abs(gap) < 1e-2


def test_gradcam_peaks_on_the_evidence():
    # Feature map = the red channel; class score = its mean. Grad-CAM must highlight the bright patch.
    model = TinyNet().eval()
    with torch.no_grad():
        model.features[0].weight.zero_()
        model.features[0].bias.zero_()
        model.features[0].weight[0, 0, 1, 1] = 1.0
        model.head.weight.zero_()
        model.head.weight[1, 0] = 1.0
    x = torch.zeros(1, 3, 32, 32)
    x[0, 0, 4:10, 20:26] = 1.0
    cam, logits = GradCAM(model, model.features)(x, 1)
    assert cam.shape == (1, 32, 32) and (cam >= 0).all()
    peak = np.unravel_index(cam[0].argmax().item(), (32, 32))
    assert 4 <= peak[0] < 10 and 20 <= peak[1] < 26


def test_gradcam_works_with_frozen_weights():
    model = TinyNet().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cam, _ = GradCAM(model, model.features)(torch.rand(2, 3, 16, 16))
    assert cam.shape == (2, 16, 16)


def test_deletion_curve_drops_when_evidence_removed():
    model = TinyNet().eval()
    with torch.no_grad():
        model.features[0].weight.zero_()
        model.features[0].bias.zero_()
        model.features[0].weight[0, 0, 1, 1] = 1.0
        model.head.weight.zero_()
        model.head.weight[1, 0] = 50.0
    x = torch.zeros(1, 3, 32, 32)
    x[0, 0, 0:8, 0:8] = 1.0
    base = torch.zeros_like(x)
    fov = np.ones((32, 32), bool)
    sal = x[0, 0].numpy()
    good, grid = cell_order(sal, fov, 8)
    rand, _ = cell_order(sal, fov, 8, np.random.default_rng(0))
    d_good = perturbation_curve(model, x, base, good, grid, 8, 1, 16, "deletion")
    d_rand = perturbation_curve(model, x, base, rand, grid, 8, 1, 16, "deletion")
    assert curve_auc(d_good) < curve_auc(d_rand)


def test_localisation_and_similarity():
    sal = np.zeros((64, 64))
    sal[10:14, 10:14] = 1
    lesion = np.zeros((64, 64), bool)
    lesion[11, 11] = True
    fov = np.ones((64, 64), bool)
    s = localisation_scores(sal, lesion, fov, tolerance_px=3)
    assert s["hit"] == 1.0 and s["mass_in_lesion"] > s["area_lesion"]
    a = np.random.default_rng(0).random((64, 64))
    assert map_similarity(a, a, fov) > 0.99
    assert np.isnan(map_similarity(a, np.zeros((64, 64)), fov))


def test_blurred_baseline_has_no_rim_artefact():
    from drxai.xai.attribution import blurred_baseline
    import cv2
    img = np.zeros((256, 256, 3), np.uint8)
    cv2.circle(img, (128, 128), 110, (180, 90, 40), -1)  # flat retina on black
    fov = img.max(2) > 0
    diff = np.abs(img.astype(float) - blurred_baseline(img, 16, fov).astype(float)).sum(2)
    assert diff[fov].max() <= 3  # flat retina (rounding only) -> baseline equals image everywhere, including the rim
    assert (blurred_baseline(img, 16, fov)[~fov] == 0).all()
