"""
Encoding Manager Service
========================
Responsible for building, caching, loading, and extending the face‑encoding
database used by the recognition pipeline.

Cache format:
    BackEnd/data/encodings/
        manifest.json
        Student1.pkl
        Student2.pkl

manifest.json stores:
{
  "students": {
    "Student1": {
      "folder_fingerprint": "...",
      "encoding_file": "Student1.pkl",
      "num_images": 5,
      "last_updated": 1620000000
    }
  }
}
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import face_recognition as fr_lib
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class EncodingManager:
    """Build, persist, and manage face‑encoding data incrementally."""

    def __init__(
        self,
        students_dir: Optional[Path] = None,
        encodings_dir: Optional[Path] = None,
        model: Optional[str] = None,
        image_extensions: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self.students_dir = students_dir or settings.STUDENTS_FACES_DIR
        self.encodings_dir = encodings_dir or settings.ENCODINGS_DIR
        self.model = model or settings.FACE_RECOGNITION_MODEL
        self.image_extensions = image_extensions or settings.SUPPORTED_IMAGE_EXTENSIONS

        self.manifest_file = self.encodings_dir / "manifest.json"

        # In‑memory cache (Phase 1 fast matching)
        self._mean_names: List[str] = []
        self._mean_encodings: List[np.ndarray] = []
        
        # Detailed cache (Phase 2 optional verification)
        self._detailed_cache: Dict[str, List[np.ndarray]] = {}

    # ── Public API ───────────────────────────────────────────────────

    def ensure_fresh(self) -> Dict[str, Any]:
        """Load the cache using manifest. Rebuild ONLY changed students."""
        self.encodings_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest()

        if not self.students_dir.exists():
            logger.warning("Students directory missing: %s", self.students_dir)
            self._clear_memory()
            return {"status": "no_dataset", "students": {}, "total_encodings": 0, "skipped": []}

        live_students = {d.name: d for d in self.students_dir.iterdir() if d.is_dir()}
        cached_students = set(manifest.get("students", {}).keys())

        rebuilt_names = []
        deleted_names = []
        skipped_files = []

        # 1. Handle deleted students (CASE D)
        for name in cached_students:
            if name not in live_students:
                self._delete_student_cache(name)
                manifest["students"].pop(name, None)
                deleted_names.append(name)

        # 2. Check each live student (CASE A, B, C)
        for name, student_dir in live_students.items():
            current_fp = self._compute_student_fingerprint(student_dir)
            cached_data = manifest.get("students", {}).get(name)

            needs_rebuild = False
            if cached_data is None:
                needs_rebuild = True  # CASE C: new student
            elif cached_data.get("folder_fingerprint") != current_fp:
                needs_rebuild = True  # CASE B: changed student

            if needs_rebuild:
                res = self._build_student_cache(name, student_dir)
                skipped_files.extend(res["skipped"])
                if res["success"]:
                    manifest.setdefault("students", {})[name] = {
                        "folder_fingerprint": current_fp,
                        "encoding_file": f"{name}.pkl",
                        "num_images": res["count"],
                        "last_updated": int(time.time()),
                    }
                    rebuilt_names.append(name)
                else:
                    # Remove if build failed completely
                    manifest.get("students", {}).pop(name, None)

        self._save_manifest(manifest)

        # 3. Preload all cached representative vectors into RAM
        self._load_all_caches(manifest)

        print("\n" + "=" * 50)
        print("Startup Cache Check:")
        print(f"Loaded cached:")
        for n in manifest.get("students", {}).keys():
            if n not in rebuilt_names and n not in deleted_names:
                print(f"- {n}")
        if rebuilt_names:
            print("Rebuilt:")
            for n in rebuilt_names:
                print(f"- {n}")
        if deleted_names:
            print("Deleted:")
            for n in deleted_names:
                print(f"- {n}")
        print(f"Total students loaded: {len(self._mean_names)}")
        print(f"Total representative vectors: {len(self._mean_encodings)}")
        print("=" * 50 + "\n")

        status = "cache_hit" if not rebuilt_names and not deleted_names else "cache_rebuilt"
        return {
            "status": status,
            "students": {n: manifest.get("students", {}).get(n, {}).get("num_images", 0) for n in manifest.get("students", {})},
            "total_encodings": sum(len(e) for e in self._detailed_cache.values()),
            "skipped": skipped_files,
        }

    def rebuild_all_encodings(self) -> Dict[str, Any]:
        """Force a full rebuild of the face-encoding cache."""
        self._clear_memory()
        
        # Delete old caches and manifest
        if self.manifest_file.exists():
            try:
                self.manifest_file.unlink()
            except OSError:
                pass
        
        for f in self.encodings_dir.glob("*.pkl"):
            try:
                f.unlink()
            except OSError:
                pass

        return self.ensure_fresh()

    def build_encodings(self) -> Dict[str, Any]:
        """Alias for rebuild_all_encodings for compatibility."""
        return self.rebuild_all_encodings()

    def load_encodings(self) -> bool:
        """Alias for ensure_fresh for compatibility."""
        self.ensure_fresh()
        return self.is_loaded

    def add_student_encoding(
        self,
        student_name: str,
        image_path: Path,
        *,
        persist: bool = True,
    ) -> bool:
        """Encode a single image and add it to the student's cache.
        Supports new student approval flow."""
        encoding = self._encode_image(image_path)
        if encoding is None:
            logger.warning("Could not encode image for %s: %s", student_name, image_path)
            return False

        # Add to detailed cache
        if student_name not in self._detailed_cache:
            self._detailed_cache[student_name] = []
        self._detailed_cache[student_name].append(encoding)

        # Update mean
        new_mean = np.mean(self._detailed_cache[student_name], axis=0)
        
        if student_name in self._mean_names:
            idx = self._mean_names.index(student_name)
            self._mean_encodings[idx] = new_mean
        else:
            self._mean_names.append(student_name)
            self._mean_encodings.append(new_mean)

        if persist:
            # Save individual pickle
            self._save_student_pickle(student_name, self._detailed_cache[student_name], new_mean)
            
            # Update manifest
            manifest = self._load_manifest()
            student_dir = self.students_dir / student_name
            current_fp = self._compute_student_fingerprint(student_dir)
            manifest.setdefault("students", {})[student_name] = {
                "folder_fingerprint": current_fp,
                "encoding_file": f"{student_name}.pkl",
                "num_images": len(self._detailed_cache[student_name]),
                "last_updated": int(time.time()),
            }
            self._save_manifest(manifest)

        logger.info("Added encoding for %s from %s", student_name, image_path)
        return True

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def names(self) -> List[str]:
        """Return the list of representative student names."""
        return self._mean_names

    @property
    def encodings(self) -> List[np.ndarray]:
        """Return the list of representative face‑encoding vectors (mean_encodings)."""
        return self._mean_encodings
        
    def get_detailed_encodings_for(self, name: str) -> List[np.ndarray]:
        return self._detailed_cache.get(name, [])

    @property
    def is_loaded(self) -> bool:
        """Check whether any encodings are available in memory."""
        return len(self._mean_encodings) > 0

    # ── Internal helpers ─────────────────────────────────────────────

    def _clear_memory(self):
        self._mean_names.clear()
        self._mean_encodings.clear()
        self._detailed_cache.clear()

    def _load_manifest(self) -> dict:
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Manifest corrupt or unreadable, starting fresh. %s", e)
        return {"students": {}}

    def _save_manifest(self, manifest: dict) -> None:
        try:
            # Atomic save
            fd, tmp_path = tempfile.mkstemp(
                prefix="manifest.",
                suffix=".tmp",
                dir=str(self.encodings_dir),
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, self.manifest_file)
        except Exception as e:
            logger.error("Failed to save manifest: %s", e)

    def _compute_student_fingerprint(self, student_dir: Path) -> str:
        """Strong hash of folder contents to detect changes."""
        records: List[str] = []
        for img_path in sorted(student_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in self.image_extensions:
                continue
            try:
                st = img_path.stat()
                records.append(f"{img_path.name}|{int(st.st_mtime)}|{st.st_size}")
            except OSError:
                continue
        if not records:
            return ""
        return hashlib.sha1("\n".join(records).encode("utf-8")).hexdigest()

    def _build_student_cache(self, student_name: str, student_dir: Path) -> dict:
        """Build encodings for a single student and save to their .pkl file."""
        encodings = []
        skipped = []
        
        image_files = sorted(
            f for f in student_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.image_extensions
        )
        
        for img_path in image_files:
            encoding = self._encode_image(img_path)
            if encoding is not None:
                encodings.append(encoding)
            else:
                skipped.append(str(img_path))
                
        if not encodings:
            return {"success": False, "count": 0, "skipped": skipped}
            
        mean_encoding = np.mean(encodings, axis=0)
        self._save_student_pickle(student_name, encodings, mean_encoding)
        return {"success": True, "count": len(encodings), "skipped": skipped}

    def _save_student_pickle(self, student_name: str, encodings: List[np.ndarray], mean_encoding: np.ndarray) -> None:
        target = self.encodings_dir / f"{student_name}.pkl"
        data = {
            "student_name": student_name,
            "encodings": encodings,
            "mean_encoding": mean_encoding,
            # We could add best_encoding here if needed
        }
        
        fd, tmp_path = tempfile.mkstemp(
            prefix=target.stem + ".",
            suffix=".tmp",
            dir=str(self.encodings_dir),
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, target)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _delete_student_cache(self, student_name: str) -> None:
        """Delete a student's .pkl file."""
        target = self.encodings_dir / f"{student_name}.pkl"
        if target.exists():
            try:
                target.unlink()
            except OSError:
                pass
        
        # Remove from memory
        if student_name in self._mean_names:
            idx = self._mean_names.index(student_name)
            self._mean_names.pop(idx)
            self._mean_encodings.pop(idx)
        self._detailed_cache.pop(student_name, None)

    def _load_all_caches(self, manifest: dict) -> None:
        """Load all valid student caches into memory."""
        self._clear_memory()
        for name, data in manifest.get("students", {}).items():
            pkl_file = self.encodings_dir / data.get("encoding_file", f"{name}.pkl")
            if not pkl_file.exists():
                logger.warning("Cache file missing for %s: %s", name, pkl_file)
                continue
                
            try:
                with open(pkl_file, "rb") as fh:
                    pkl_data = pickle.load(fh)
                    
                mean_enc = pkl_data.get("mean_encoding")
                encs = pkl_data.get("encodings", [])
                
                if mean_enc is not None:
                    self._mean_names.append(name)
                    self._mean_encodings.append(mean_enc)
                    self._detailed_cache[name] = encs
            except Exception as e:
                logger.error("Failed to load cache for %s: %s", name, e)

    def _encode_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load an image, validate quality, and return its first face encoding, or None."""
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            logger.error("Could not read image: %s", image_path)
            return None

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance < settings.MIN_FACE_SHARPNESS:
            logger.warning("Skipped %s: Image too blurry (variance: %.2f)", image_path.name, variance)
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        locations = fr_lib.face_locations(img_rgb, model=self.model)
        
        if len(locations) == 0:
            logger.warning("Skipped %s: No faces found", image_path.name)
            return None
        if len(locations) > 1:
            logger.warning("Skipped %s: Multiple faces found (%d)", image_path.name, len(locations))
            return None

        top, right, bottom, left = locations[0]
        face_w = right - left
        face_h = bottom - top
        if face_w < settings.MIN_FACE_SIZE or face_h < settings.MIN_FACE_SIZE:
            logger.warning("Skipped %s: Face too small (%dx%d)", image_path.name, face_w, face_h)
            return None

        encodings = fr_lib.face_encodings(img_rgb, known_face_locations=locations, num_jitters=settings.NUM_JITTERS)

        if not encodings:
            return None
        return encodings[0]
