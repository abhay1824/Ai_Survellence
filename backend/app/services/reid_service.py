"""
Eagle Eye Cross-Camera Person Re-Identification (Re-ID) Engine
Performs real-time appearance feature extraction and multi-camera identity matching
using spatial color histograms and cosine similarity.
"""

import math
import time
from typing import Dict, List, Any, Optional, Tuple

class ReIDService:
    # Global feature gallery storing {global_id: {"vector": [...], "camera_id": str, "last_seen": float, "track_id": int}}
    gallery: Dict[str, Dict[str, Any]] = {}
    next_global_id: int = 1
    similarity_threshold: float = 0.75

    @classmethod
    def extract_feature_vector(cls, image_crop: Any, bbox: Tuple[float, float, float, float]) -> List[float]:
        """
        Extracts lightweight normalized HSV spatial color histogram & aspect ratio embedding.
        If image crop is numpy array, extracts HSV histogram. Otherwise generates synthetic feature vector based on aspect ratio & position.
        """
        w = max(bbox[2] - bbox[0], 1.0)
        h = max(bbox[3] - bbox[1], 1.0)
        aspect_ratio = w / h

        try:
            import cv2
            import numpy as np
            if isinstance(image_crop, np.ndarray) and image_crop.size > 0:
                hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
                # Compute 8-bin Hue, 4-bin Saturation, 4-bin Value histogram
                hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
                hist_s = cv2.calcHist([hsv], [1], None, [4], [0, 256]).flatten()
                hist_v = cv2.calcHist([hsv], [2], None, [4], [0, 256]).flatten()
                
                raw_vec = np.concatenate([hist_h, hist_s, hist_v, [aspect_ratio]])
                norm = np.linalg.norm(raw_vec)
                if norm > 0:
                    raw_vec = raw_vec / norm
                return raw_vec.tolist()
        except ImportError:
            pass

        # Fallback math embedding based on bounding box geometry
        vector = [
            math.sin(aspect_ratio),
            math.cos(aspect_ratio),
            w / 100.0,
            h / 100.0,
            (bbox[0] + bbox[2]) / 1000.0,
            (bbox[1] + bbox[3]) / 1000.0
        ]
        norm = math.sqrt(sum(x*x for x in vector))
        return [x / norm for x in vector] if norm > 0 else vector

    @classmethod
    def cosine_similarity(cls, vec1: List[float], vec2: List[float]) -> float:
        """Calculates cosine similarity between two normalized vectors."""
        if len(vec1) != len(vec2):
            # Pad or truncate to match lengths
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    @classmethod
    def match_or_register(
        cls,
        camera_id: str,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        image_crop: Any = None
    ) -> Tuple[str, float]:
        """
        Matches a person detection against global Re-ID gallery across all cameras.
        Returns tuple of (global_id_str, match_confidence_score).
        """
        feat_vec = cls.extract_feature_vector(image_crop, bbox)
        now = time.time()

        best_gid = None
        best_sim = 0.0

        for gid, entry in cls.gallery.items():
            # Skip if entry expired (> 15 minutes)
            if now - entry["last_seen"] > 900:
                continue

            sim = cls.cosine_similarity(feat_vec, entry["vector"])
            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        if best_sim >= cls.similarity_threshold and best_gid is not None:
            # Update existing global identity with moving average feature vector
            old_vec = cls.gallery[best_gid]["vector"]
            updated_vec = [0.8 * o + 0.2 * n for o, n in zip(old_vec, feat_vec)]
            cls.gallery[best_gid] = {
                "vector": updated_vec,
                "camera_id": camera_id,
                "last_seen": now,
                "track_id": track_id
            }
            return best_gid, round(best_sim, 2)
        else:
            # Register new global identity
            new_gid = f"GLOBAL-ID-{cls.next_global_id:03d}"
            cls.next_global_id += 1
            cls.gallery[new_gid] = {
                "vector": feat_vec,
                "camera_id": camera_id,
                "last_seen": now,
                "track_id": track_id
            }
            return new_gid, 1.00
