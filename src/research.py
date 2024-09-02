import matplotlib.pyplot as pl
import numpy as np
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.segmentation import mark_boundaries
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from Models.CNN_model import CNN
from Models.VGG19_model import VGG19

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cnn = CNN()
cnn.load_state_dict(torch.load("assets/cnn_model.pth", map_location=device))
cnn.eval()

original_vgg19_model = models.vgg19(weights="VGG19_Weights.DEFAULT")
vgg = VGG19(original_vgg19_model)
vgg.load_state_dict(torch.load("assets/vgg19_transfer.pth", map_location=device))
vgg.eval()


def explain_with_lime(model: nn.Module, image_path: str, target_class: int) -> None:
    image_transform = transforms.ToTensor()

    image = np.array(Image.open(image_path).convert("RGB").resize((224, 224)))

    def _batch_predict(images: np.ndarray) -> np.ndarray:
        # A function that takes a batch of images and returns a prediction for each image.
        batch = torch.stack(tuple(image_transform(i) for i in images), dim=0)
        batch = batch.to(device)
        logits = model(batch)
        probabilities = F.softmax(logits, dim=1)
        return probabilities.detach().numpy()

    explainer = lime_image.LimeImageExplainer()
    segmenter = SegmentationAlgorithm("quickshift")
    explanation = explainer.explain_instance(
        image=image,
        classifier_fn=_batch_predict,
        segmentation_fn=segmenter,
        num_samples=100,
    )
    image, mask = explanation.get_image_and_mask(
        label=target_class, positive_only=False, num_features=2
    )
    marked_image = mark_boundaries(image / 255.0, mask)
    pl.figure(figsize=(10, 10))
    pl.imshow(marked_image)
    pl.axis("off")
    pl.show()


def explain_with_grad_cam(model: nn.Module, image_path: str, target_class: int) -> None:
    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(image_path)
    img = img.convert("RGB")
    img = img.resize((224, 224))

    input_tensor: torch.Tensor = image_transform(img).unsqueeze(0)
    target_layer: nn.Conv2d = [cnn.features[6]]  # Last Conv2d layer

    targets = [ClassifierOutputTarget(target_class)]
    cam = GradCAM(model=model, target_layers=target_layer)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(np.array(img) / 255, grayscale_cam, use_rgb=True)

    pl.imshow(visualization)
    pl.axis("off")
    pl.show()


def explain_with_shap(model: nn.Module, image_path: str):
    image_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    def _prepare_image(path: str):
        image_to_explain = Image.open(path)
        image_to_explain = image_to_explain.convert("RGB")
        image_to_explain = image_transform(image_to_explain)
        image_to_explain = image_to_explain.unsqueeze(0)
        image_to_explain = image_to_explain.to(device)
        return image_to_explain

    test_data = ImageFolder("assets/Dataset/Testing", transform=image_transform)
    test_loader = DataLoader(test_data, batch_size=200, shuffle=False)
    batch = next(iter(test_loader))
    background, _ = batch
    background = background[:10]
    image = _prepare_image(image_path)
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(image)
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(image.numpy(), 1, -1), 1, 2)
    labels = ["glejak", "guz opon mózgowych", "brak nowotworu", "guz przysadki"]
    shap.image_plot(shap_numpy, test_numpy, labels)


# explain_with_lime(cnn, "assets/test_meningioma.jpg", 1)
explain_with_grad_cam(cnn, "assets/test_meningioma.jpg", 1)
