# ============================================================
# BLOCK 3: INFERENCE + SIMILARITY SCORING
# ============================================================

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── Shared constants ──
BEST_MODEL_PATH  = "/kaggle/input/datasets/yuganjain/modelsspec/best_model.pth"
EMBEDDINGS_PATH  = "/kaggle/working/reference_embeddings.npz"

POSE_DIM         = 33 * 4
EFFICIENTNET_DIM = 1280
IMG_SIZE         = 224
NUM_FRAMES       = 16

CLASSES = ['cover', 'defense', 'flick', 'hook', 'latecut',
           'lofted', 'pull', 'square_cut', 'straight', 'sweep']
NUM_CLASSES = len(CLASSES)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# MODEL DEFINITION
# ============================================================

class CricketModel(nn.Module):
    def __init__(self):
        super().__init__()
        eff            = efficientnet_b0(weights=None)
        self.backbone  = nn.Sequential(*list(eff.children())[:-1])
        self.vis_proj  = nn.Linear(1280, 256)
        self.pose_proj = nn.Linear(POSE_DIM, 256)
        self.lstm      = nn.LSTM(512, 512, batch_first=True, bidirectional=True)
        self.cls       = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, NUM_CLASSES)
        )

    def forward(self, x, pose):
        B, T, C, H, W = x.shape
        feat = self.backbone(x.view(B * T, C, H, W)).view(B, T, 1280)
        v    = self.vis_proj(feat)
        p    = self.pose_proj(pose)
        x    = torch.cat([v, p], dim=-1)
        x, _ = self.lstm(x)
        x    = x.mean(dim=1)
        return self.cls(x)

    def embed_frames(self, x, pose):
        """
        Returns pre-LSTM per-frame embeddings as numpy array (T, 512).
        Stops before the LSTM so single-frame and multi-frame embeddings
        live in the same feature space — required for image↔video comparison.
        FIX 1: Added missing 'self'
        FIX 2: Stop before LSTM (512-d, not 1024-d)
        FIX 3: Returns numpy directly so caller needs no extra .cpu().numpy()
        """
        B, T, C, H, W = x.shape
        feat = self.backbone(x.view(B * T, C, H, W)).view(B, T, 1280)
        v    = self.vis_proj(feat)          # (B, T, 256)
        p    = self.pose_proj(pose)         # (B, T, 256)
        emb  = torch.cat([v, p], dim=-1)    # (B, T, 512)
        return emb.squeeze(0).cpu().numpy() # (T, 512)  — B=1 always here

# ============================================================
# POSE EXTRACTOR
# ============================================================

class PoseExtractor:
    def __init__(self):
        try:
            base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Pose model failed: {e}")
            self.detector = None

    def extract(self, frame_rgb: np.ndarray) -> np.ndarray:
        if self.detector is None:
            return np.zeros(POSE_DIM, dtype=np.float32)
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result   = self.detector.detect(mp_image)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                vec = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
                    dtype=np.float32
                ).flatten()
                lh, rh = landmarks[23], landmarks[24]
                cx, cy = (lh.x + rh.x) / 2, (lh.y + rh.y) / 2
                for i in range(33):
                    vec[i * 4]     -= cx
                    vec[i * 4 + 1] -= cy
                return vec
            return np.zeros(POSE_DIM, dtype=np.float32)
        except Exception:
            return np.zeros(POSE_DIM, dtype=np.float32)

# ============================================================
# UTILITIES
# ============================================================

video_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_trained_model(ckpt_path: str) -> CricketModel:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = CricketModel().to(device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")
    return model


def load_reference_embeddings(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    data = np.load(path, allow_pickle=False)
    embs = {k: data[k] for k in data.files}
    print(f"Loaded reference embeddings for classes: {list(embs.keys())}")
    # Sanity check: warn if ref embeddings are 1024-d (built with old Block 2)
    for cls, arr in embs.items():
        if arr.shape[-1] != 512:
            print(f"  WARNING: '{cls}' embeddings are {arr.shape[-1]}-d, expected 512-d. "
                  "Re-run Block 2 with the pre-LSTM fix.")
    return embs


def sample_video_frames(video_path: str,
                         pose_extractor: PoseExtractor,
                         num_frames: int = NUM_FRAMES):
    """
    Sample `num_frames` uniformly from a video.
    Returns:
        frame_tensor : (1, T, 3, H, W)
        pose_tensor  : (1, T, POSE_DIM)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, max(total - 1, 1), num_frames, dtype=int)

    frames, poses = [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose = pose_extractor.extract(rgb)
            poses.append(torch.from_numpy(pose))
            frames.append(video_tf(rgb))
        else:
            frames.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
            poses.append(torch.zeros(POSE_DIM))

    cap.release()

    frame_tensor = torch.stack(frames).unsqueeze(0).to(device)  # (1, T, 3, H, W)
    pose_tensor  = torch.stack(poses).unsqueeze(0).to(device)   # (1, T, POSE_DIM)
    return frame_tensor, pose_tensor

# ============================================================
# SIMILARITY FUNCTIONS
# ============================================================

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (M, D), b: (N, D) → (M, N) pairwise cosine similarities."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def euclidean_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (M, D), b: (N, D) → (M, N) pairwise Euclidean distances."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def aggregate_similarity(video_embs: np.ndarray,
                          ref_embs:   np.ndarray,
                          metric: str = 'cosine') -> float:
    """
    For each video frame, find its BEST matching reference embedding,
    then average those best-match scores across all frames.
    video_embs : (T, 512)
    ref_embs   : (N, 512)
    """
    if metric == 'cosine':
        sim_matrix     = cosine_similarity_matrix(video_embs, ref_embs)  # (T, N)
        best_per_frame = sim_matrix.max(axis=1)                           # (T,)
        return float(best_per_frame.mean())
    elif metric == 'euclidean':
        dist_matrix    = euclidean_distance_matrix(video_embs, ref_embs) # (T, N)
        best_per_frame = dist_matrix.min(axis=1)                          # (T,)
        return float(-best_per_frame.mean())   # negate: higher = more similar
    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose 'cosine' or 'euclidean'.")

# ============================================================
# MAIN INFERENCE FUNCTION
# ============================================================

def classify_and_score(video_path:     str,
                        model:          CricketModel,
                        pose_extractor: PoseExtractor,
                        ref_embeddings: dict,
                        metric: str = 'cosine') -> dict:
    """
    Full inference pipeline for a single video.
    Steps:
      1. Sample frames from video
      2. Classify → predicted class + confidence
      3. Extract pre-LSTM per-frame embeddings  (T, 512)
      4. Compare against reference embeddings   (N, 512)
      5. Return best-match-averaged similarity score
    """
    print(f"\nProcessing: {os.path.basename(video_path)}")

    # ── STEP 1: Sample frames ──
    frame_tensor, pose_tensor = sample_video_frames(video_path, pose_extractor)

    # ── STEP 2: Classification (uses full forward pass with LSTM) ──
    with torch.no_grad():
        logits     = model(frame_tensor, pose_tensor)         # (1, NUM_CLASSES)
        probs      = torch.softmax(logits, dim=1).squeeze(0)  # (NUM_CLASSES,)
        class_idx  = int(probs.argmax().item())
        confidence = float(probs[class_idx].item())
        pred_class = CLASSES[class_idx]

    print(f"  Predicted class : {pred_class}  (confidence: {confidence:.3f})")

    # ── STEP 3: Pre-LSTM per-frame embeddings ──
    # FIX: embed_frames now returns numpy (T, 512) directly — no extra .cpu().numpy()
    with torch.no_grad():
        video_embs = model.embed_frames(frame_tensor, pose_tensor)  # (T, 512)

    print(f"  Video embeddings shape: {video_embs.shape}")

    # ── STEP 4: Similarity scoring ──
    if pred_class not in ref_embeddings:
        print(f"  WARNING: No reference embeddings found for '{pred_class}'. "
              "Score will be NaN.")
        sim_score = float('nan')
    else:
        ref_embs  = ref_embeddings[pred_class]  # (N, 512)
        sim_score = aggregate_similarity(video_embs, ref_embs, metric=metric)
        print(f"  Similarity score : {sim_score:.4f}  (metric: {metric})")

    return {
        'predicted_class':  pred_class,
        'class_index':      class_idx,
        'confidence':       confidence,
        'similarity_score': sim_score,
        'metric':           metric,
    }

# ============================================================
# BATCH INFERENCE
# ============================================================

def batch_infer(video_paths:    list,
                model:          CricketModel,
                pose_extractor: PoseExtractor,
                ref_embeddings: dict,
                metric: str = 'cosine') -> list[dict]:
    """Run inference on a list of video files and return a list of result dicts."""
    results = []
    for vp in tqdm(video_paths, desc="Inferring"):
        try:
            res = classify_and_score(vp, model, pose_extractor, ref_embeddings, metric)
            res['video'] = vp
        except Exception as e:
            print(f"  ERROR processing {vp}: {e}")
            res = {'video': vp, 'error': str(e)}
        results.append(res)
    return results

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BLOCK 3 — Inference + Similarity Scoring")
    print("=" * 60 + "\n")

    model          = load_trained_model(BEST_MODEL_PATH)
    ref_embeddings = load_reference_embeddings(EMBEDDINGS_PATH)
    pose_extractor = PoseExtractor()

    video_path = "/kaggle/input/datasets/yuganjain/cricket-shot/cricket-shot/cricketshot/cricketshot/test/pull/pull_0004.avi"

    result = classify_and_score(
        video_path     = video_path,
        model          = model,
        pose_extractor = pose_extractor,
        ref_embeddings = ref_embeddings,
        metric         = 'cosine',
    )

    print("\n" + "─" * 40)
    print("  FINAL RESULT")
    print("─" * 40)
    print(f"  Predicted class  : {result['predicted_class']}")
    print(f"  Confidence       : {result['confidence']:.4f}")
    print(f"  Similarity score : {result['similarity_score']:.4f}  ({result['metric']})")
    print("─" * 40)