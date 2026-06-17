import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import cv2
from sewar.full_ref import vifp
from scipy.stats import entropy
from PIL import Image
import argparse
from tqdm import tqdm

# Custom Dataset
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Transform for Inception
inception_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

# Calculation Functions
def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    assert sigma1.shape[0] == sigma1.shape[1], "sigma1 is not square"
    assert sigma2.shape[0] == sigma2.shape[1], "sigma2 is not square"

    ssdif = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdif + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_mssim(img1, img2):
    C1 = 6.5025
    C2 = 58.5225
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def calculate_vif(img1, img2):
    vif_values = [vifp(img1[:, :, i], img2[:, :, i]) for i in range(3)]
    return np.mean(vif_values)
# 从加载器中提取图像并计算Inception模型的激活
def extract_activations(data_loader, model, device):
    model.eval()
    activations = []
    device = torch.device(device)

    with torch.no_grad():
        for images in data_loader:
            images = images.to(device)
            features = model(images)
            activations.append(features.cpu().numpy())

    return np.concatenate(activations)

def calculate_inception_scores(image_paths, device, batch_size, local_inception_v3_path,resize=True, splits=10):
    # inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model = inception_v3(pretrained=False, transform_input=False).to(device)
    # 然后手动加载权重
    model_url = "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth"
    if not os.path.exists(local_inception_v3_path):
        # 如果还没有权重文件，创建目录并下载
        os.makedirs(os.path.dirname(local_inception_v3_path), exist_ok=True)
        torch.hub.download_url_to_file(model_url, local_inception_v3_path)

    inception_model.load_state_dict(
        torch.load(local_inception_v3_path, map_location=device)
    )
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return torch.nn.functional.softmax(x, dim=1).data.cpu().numpy()

    N = len(image_paths)
    preds = np.zeros((N, 1000))

    for i in tqdm(range(0, N, batch_size)):
        start, end = i, i + batch_size
        images = np.array([np.asarray(Image.open(p).convert('RGB')).astype(np.float32) for p in image_paths[start:end]])
        images = images.transpose((0, 3, 1, 2)) / 255
        batch = torch.from_numpy(images).type(torch.FloatTensor).to(device)
        
        preds[start:end] = get_pred(batch)

    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = [entropy(part[i, :], py) for i in range(part.shape[0])]
        split_scores.append(np.exp(scores))
    
    return np.max(split_scores), np.mean(split_scores)

# fid avg_mssim avg_vif max_is avg_is
def calculate(real_images, generated_images, batch_size=8, device='cuda', local_inception_v3_path ="/root/shuqian/checkpoints/inception_v3_google-0cc3c7bd.pth"):
    metrics = {}
    # Load datasets
    real_dataset = CustomImageDataset(real_images, transform=inception_transform)
    generated_dataset = CustomImageDataset(generated_images, transform=inception_transform)

    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    gen_loader = DataLoader(generated_dataset, batch_size=batch_size, shuffle=False)

    # FID Calculation
    print("Calculating FID...")
    # inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model = inception_v3(pretrained=False, transform_input=False).to(device)
    # 然后手动加载权重
    model_url = "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth"
    if not os.path.exists(local_inception_v3_path):
        # 如果还没有权重文件，创建目录并下载
        os.makedirs(os.path.dirname(local_inception_v3_path), exist_ok=True)
        torch.hub.download_url_to_file(model_url, local_inception_v3_path)

    inception_model.load_state_dict(
        torch.load(local_inception_v3_path, map_location=device)
    )

    inception_model.fc = nn.Identity()
    real_activations = extract_activations(real_loader, inception_model, device)
    gen_activations = extract_activations(gen_loader, inception_model, device)
    fid_value = calculate_fid(real_activations, gen_activations)
    print('FID:', fid_value)
    metrics['fid'] = fid_value

    # MSSIM and VIF Calculation
    print("Calculating MSSIM and VIF...")
    mssim_values, vif_values = [], []
    files = os.listdir(generated_images)
    for file in files:
        gen_path = os.path.join(generated_images, file)
        gt_path = os.path.join(real_images, file)

        if not os.path.isfile(gen_path) or not os.path.isfile(gt_path):
            print(f"File {file} not found in both directories. Skipping.")
            continue

        img_gen = cv2.imread(gen_path, cv2.IMREAD_GRAYSCALE)
        img_real = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        mssim_values.append(calculate_mssim(img_gen, img_real))

        img_gen_color = cv2.imread(gen_path)
        img_real_color = cv2.imread(gt_path)
        vif_values.append(calculate_vif(img_gen_color, img_real_color))

    avg_mssim = np.mean(mssim_values)
    avg_vif = np.mean(vif_values)
    print(f"Average MSSIM for all images: {avg_mssim}")
    print(f"Average VIF for all images: {avg_vif}")
    metrics['avg_mssim'] = avg_mssim
    metrics['avg_vif'] = avg_vif

    # # IS Calculation
    # print("Calculating Inception Score...")
    # image_files = [os.path.join(generated_images, f) for f in files]
    # max_is, avg_is = calculate_inception_scores(image_files, device, batch_size,local_inception_v3_path=local_inception_v3_path)
    # print('MAX IS is %.4f' % max_is)
    # print('The average IS is %.4f' % avg_is)
    # metrics['max_is'] = max_is
    # metrics['avg_is'] = avg_is

    return metrics

# Main script
if __name__ == "__main__":
    calculate("/root/shuqian/dataset/compare128/photo/xm2vts/decp",
              "/root/shuqian/dataset/compare128/photo/xm2vts/gt")