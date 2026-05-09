"""
Encoding Manager Service
========================
Responsible for building, caching, loading, and extending the face‑encoding
database used by the recognition pipeline.

Cache format  (``face_encodings.pkl``):
    {
        "names":     List[str],           # parallel lists
        "encodings": List[np.ndarray],    # 128‑d face descriptors
    }
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class EncodingManager:
    """Build, persist, and manage face‑encoding data."""

    def __init__(
        self,
        students_dir: Optional[Path] = None,
        encodings_file: Optional[Path] = None,
        model: Optional[str] = None,
        image_extensions: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self.students_dir = students_dir or settings.STUDENTS_FACES_DIR
        self.encodings_file = encodings_file or settings.ENCODINGS_FILE
        self.model = model or settings.FACE_RECOGNITION_MODEL
        self.image_extensions = image_extensions or settings.SUPPORTED_IMAGE_EXTENSIONS

        # In‑memory cache
        self._names: List[str] = []
        self._encodings: List[np.ndarray] = []

    # ── Public API ───────────────────────────────────────────────────

    def build_encodings(self) -> Dict[str, Any]:
        """Scan ``students_dir``, encode all faces, and persist the cache.

        Returns:
            Summary dict with per‑student counts and any skipped images.
        """
        self._names.clear()
        self._encodings.clear()
        summary: Dict[str, Any] = {"students": {}, "total_encodings": 0, "skipped": []}

        if not self.students_dir.exists():
            raise FileNotFoundError(
                f"Students directory not found: {self.students_dir}"
            )

        student_dirs = sorted(
            [d for d in self.students_dir.iterdir() if d.is_dir()]
        )
        if not student_dirs:
            logger.warning("No student subdirectories found in %s", self.students_dir)
            return summary

        for student_dir in student_dirs:
            student_name = student_dir.name
            count = 0

            image_files = sorted(
                f for f in student_dir.iterdir()
                if f.is_file() and f.suffix.lower() in self.image_extensions
            )
            if not image_files:
                logger.warning("No images found for student: %s", student_name)
                continue

            for img_path in image_files:
                encoding = self._encode_image(img_path)
                if encoding is not None:
                    self._names.append(student_name)
                    self._encodings.append(encoding)
                    count += 1
                else:
                    summary["skipped"].append(str(img_path))
                    logger.warning("Skipped (no face found): %s", img_path)

            summary["students"][student_name] = count
            logger.info("Encoded %d images for %s", count, student_name)

        summary["total_encodings"] = len(self._encodings)
        self._save_cache()
        logger.info(
            "Encoding build complete – %d encodings for %d students",
            len(self._encodings),
            len(summary["students"]),
        )
        return summary

    def load_encodings(self) -> bool:
        """Load encodings from the persisted ``.pkl`` cache.

        Returns:
            ``True`` if cache loaded successfully, ``False`` otherwise.
        """
        if not self.encodings_file.exists():
            logger.warning("No encoding cache found at %s", self.encodings_file)
            return False

        try:
            with open(self.encodings_file, "rb") as fh:
                data: Dict[str, Any] = pickle.load(fh)
            self._names = data["names"]
            self._encodings = data["encodings"]
            logger.info(
                "Loaded %d encodings from cache (%s)",
                len(self._encodings),
                self.encodings_file,
            )
            return True
        except (pickle.UnpicklingError, KeyError, EOFError) as exc:
            logger.error("Failed to load encoding cache: %s", exc)
            return False

    def add_student_encoding(
        self,
        student_name: str,
        image_path: Path,
        *,
        persist: bool = True,
    ) -> bool:
        """Encode a single image and append it to the cache.

        This supports the future unknown→registration flow where new
        students are approved by an admin and their encodings are
        incrementally added without a full rebuild.

        Args:
            student_name: The student's folder / display name.
            image_path:   Path to a single face image.
            persist:      Whether to immediately save the updated cache.

        Returns:
            ``True`` if the encoding was added successfully.
        """
        encoding = self._encode_image(image_path)
        if encoding is None:
            logger.warning(
                "Could not encode image for %s: %s", student_name, image_path
            )
            return False

        self._names.append(student_name)
        self._encodings.append(encoding)

        if persist:
            self._save_cache()

        logger.info("Added encoding for %s from %s", student_name, image_path)
        return True

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def names(self) -> List[str]:
        """Return the list of student names (parallel to encodings)."""
        return self._names

    @property
    def encodings(self) -> List[np.ndarray]:
        """Return the list of face‑encoding vectors."""
        return self._encodings

    @property
    def is_loaded(self) -> bool:
        """Check whether any encodings are available in memory."""
        return len(self._encodings) > 0

    # ── Internal helpers ─────────────────────────────────────────────

    def _encode_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load an image and return its first face encoding, or ``None``."""
        # Use cv2 to read → convert to RGB (face_recognition expects RGB)
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            logger.error("Could not read image: %s", image_path)
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)

        if not encodings:
            return None
        return encodings[0]

    def _save_cache(self) -> None:
        """Persist current in‑memory encodings to disk."""
        self.encodings_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "names": self._names,
            "encodings": self._encodings,
        }
        with open(self.encodings_file, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved %d encodings to %s", len(self._encodings), self.encodings_file)
