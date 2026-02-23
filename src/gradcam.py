import torch
import torch.nn.functional as F


class FasterRCNNGradCAM:
    """GradCAM for Faster RCNN ResNet50 FPN — targets backbone.body.layer4."""

    def __init__(self, model, target_layer_name="backbone.body.layer4.2.conv3"):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hooks = []

        target_layer = self._get_layer(target_layer_name)
        self._hooks.append(target_layer.register_forward_hook(self._save_activation))
        self._hooks.append(target_layer.register_full_backward_hook(self._save_gradient))

    def _get_layer(self, name):
        layer = self.model
        for part in name.split("."):
            layer = getattr(layer, part)
        return layer

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def compute(self, image_tensor, device, threshold=0.5, target_class=None):
        """
        Returns:
            filtered_preds : dict(boxes, labels, scores) or None
            heatmaps       : list of np.array (H, W) in [0, 1]
        """
        self.model.eval()
        image = image_tensor.unsqueeze(0).to(device)

        with torch.enable_grad():
            predictions = self.model(image)[0]

        keep = predictions['scores'] > threshold
        if target_class is not None:
            keep = keep & (predictions['labels'] == target_class)

        boxes  = predictions['boxes'][keep]
        labels = predictions['labels'][keep]
        scores = predictions['scores'][keep]

        if len(scores) == 0:
            return None, []

        heatmaps = []
        for i in range(len(scores)):
            self.model.zero_grad()
            self.gradients = None

            scores[i].backward(retain_graph=(i < len(scores) - 1))

            if self.gradients is None:
                heatmaps.append(None)
                continue

            weights = self.gradients.mean(dim=(2, 3))[0]          # (C,)
            cam = (weights[:, None, None] * self.activations[0]).sum(dim=0)  # (H, W)
            cam = F.relu(cam)

            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)

            heatmaps.append(cam.cpu().numpy())

        return {'boxes': boxes.detach(), 'labels': labels.detach(), 'scores': scores.detach()}, heatmaps