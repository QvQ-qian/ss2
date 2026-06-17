from deepface import DeepFace
import os
import numpy as np
from tqdm import tqdm
import argparse

# --- 配置区域，修改为你的路径和度量方式 ---
GALLERY_DIR     = '/root/xuegui/calculate/CUHK_GT/'
PROBE_DIR       = '/root/xuegui/calculate/Generated_CUHK/output_cuhk_veir/'   # 或
MODEL_NAME      = 'SFace'
DISTANCE_METRIC = 'cosine'               # 'cosine', 'euclidean', 'euclidean_l2', ...
TOP_K           = 5
FARs            = [0.001, 0.01]          # 0.1% 和 1%

# --- 工具函数 ---
def list_images(folder):
    return sorted([os.path.join(folder, f)
                   for f in os.listdir(folder)
                   if f.lower().endswith(('.jpg', '.png'))])

def id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]

def compute_embeddings(paths, model_name, detector_backend='opencv', enforce_detection=False):
    """用 DeepFace.represent 计算所有图片的 embedding"""
    embs = {}
    for p in tqdm(paths, desc=f"Embedding {len(paths)}"):
        resp = DeepFace.represent(
            img_path=p,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
        )
        embs[p] = np.array(resp[0]['embedding'])
    return embs

def compute_distance(vec1, vec2, metric):
    """基于 metric 计算两向量间距离"""
    if metric == 'cosine':
        sim = vec1 @ vec2 / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return 1 - sim
    elif metric in ('euclidean', 'euclidean_l2'):
        return np.linalg.norm(vec1 - vec2)
    else:
        # 回退到 DeepFace.verify（稍慢），但支持所有内置 metric
        return DeepFace.verify(
            img1_path=None, img2_path=None,
            model_name=MODEL_NAME,
            distance_metric=metric,
            enforce_detection=False,
            # 直接传 embedding 会忽略路径，但 DeepFace API 不支持纯向量输入
        ).get('distance')

def evaluate_rank(probe_embs, gallery_embs, k, metric):
    gallery_paths = list(gallery_embs.keys())
    gallery_ids   = [id_from_path(p) for p in gallery_paths]
    G = np.stack([gallery_embs[p] for p in gallery_paths])

    correct = 0
    for probe_path, vec in probe_embs.items():
        # 逐一计算距离
        dists = np.array([compute_distance(vec, gallery_embs[g], metric)
                          for g in gallery_paths])
        topk = np.argsort(dists)[:k]
        if id_from_path(probe_path) in {gallery_ids[i] for i in topk}:
            correct += 1
    return correct / len(probe_embs)

def compute_vr(probe_embs, gallery_embs, fars, metric):
    gallery_paths = list(gallery_embs.keys())
    gallery_ids   = [id_from_path(p) for p in gallery_paths]

    genuine, imposter = [], []
    for probe_path, vec in probe_embs.items():
        pid = id_from_path(probe_path)
        dists = np.array([compute_distance(vec, gallery_embs[g], metric)
                          for g in gallery_paths])
        for d, gid in zip(dists, gallery_ids):
            (genuine if gid == pid else imposter).append(d)

    genuine = np.array(genuine)
    imposter = np.array(imposter)
    imp_sorted = np.sort(imposter)
    n_imp = len(imp_sorted)

    results = {}
    for far in fars:
        idx = min(int(np.floor(far * n_imp)), n_imp - 1)
        thr = imp_sorted[idx]
        vr  = np.mean(genuine <= thr)
        results[far] = (thr, vr)
    return results

# rank_1_accuracy  rank_5_accuracy
def calculate(gallery,probe,model='SFace',distance_metric='cosine',top_k=5,fars=[0.001, 0.01]):
    metrics = {}
    gallery_paths = list_images(gallery)
    probe_paths = list_images(probe)

    print("Computing embeddings...")
    gallery_embs = compute_embeddings(gallery_paths, model)
    probe_embs = compute_embeddings(probe_paths, model)


    print("\n→ Rank-1 Accuracy:")
    rank_1 = evaluate_rank(probe_embs, gallery_embs, k=1, metric=distance_metric)
    print(f"  {rank_1:.2%}")
    metrics['rank_1_accuracy'] = rank_1

    print("\n→ Rank-5 Accuracy:")
    rank_5 = evaluate_rank(probe_embs, gallery_embs, k=top_k, metric=distance_metric)
    print(f"  {rank_5:.2%}")
    metrics['rank_5_accuracy'] = rank_5

    # print("\n→ Verification Rate @ FAR:")
    # vr_results = compute_vr(probe_embs, gallery_embs, fars, distance_metric)
    # for far, (thr, vr) in vr_results.items():
    #     print(f"  FAR = {far * 100:.1f}%: threshold = {thr:.4f}, VR = {vr:.2%}")

    return metrics

if __name__ == '__main__':
    os.environ['DEEPFACE_HOME'] = '/root/shuqian/checkpoints'
    calculate("/root/shuqian/dataset/compare128/photo/xm2vts/gt",
              "/root/shuqian/dataset/compare128/photo/xm2vts/decp")