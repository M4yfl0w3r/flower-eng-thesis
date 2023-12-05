from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

train_data_path = Path('../assets/Dataset/Training')
train_images_path = list(Path('../assets/Dataset/Training').glob('*/*jpg'))

image_transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean = [0.1855, 0.1855, 0.1855], std = [0.2003, 0.2003, 0.2004])
])

train_data = ImageFolder(train_data_path, transform = image_transform)
train_loader = DataLoader(train_data, batch_size = len(train_images_path), shuffle = True)

images, _ = next(iter(train_loader))
mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
print(f'Mean = {mean} | Std = {std}')