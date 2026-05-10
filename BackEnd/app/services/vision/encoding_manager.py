"""
Encoding Manager Service
========================
Responsible for building, caching, loading, and extending the face‑encoding
database used by the recognition pipeline.

Cache format  (``face_encodings.pkl``):
    {
        "names":       List[str],           # parallel lists
        "encodings":   List[np.ndarray],    # 128‑d face descriptors
        "fingerprint": str,                 # SHA‑1 of dataset state
    }

Dataset fingerprint
-------------------
The cache embeds a hash of every image's relative path + mtime + size.
On startup, :meth:`EncodingManager.ensure_fresh` compares the on-disk
fingerprint to the live dataset.  If they match, the cache is loaded
in milliseconds and recognition is immediately ready.  If the dataset
has changed (image added, deleted, replaced, or modified), the cache
is rebuilt automatically — once — and re-saved with the new
fingerprint.

This eliminates the ~30-60 s per-startup rebuild that
``VisionSession`` used to force unconditionally.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import face_recognition as fr_lib
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
        self._fingerprint: Optional[str] = None

    # ── Public API ───────────────────────────────────────────────────

    def ensure_fresh(self) -> Dict[str, Any]:
        """Load the cache if its fingerprint matches the live dataset;
        otherwise rebuild it from disk.

        This is the recommended entry point for the application
        startup path — it avoids the 30-60 s per-launch rebuild while
        still guaranteeing that deleted students cannot be recognised
        and newly-added students take effect immediately.

        Returns a summary dict::

            {
                "status":           "cache_hit" | "cache_rebuilt" | "no_dataset",
                "students":         {name: count, ...},
                "total_encodings":  int,
                "fingerprint":      str,
                "skipped":          [paths],
            }
        """
        live_fp = self._compute_dataset_fingerprint()

        # Empty / missing dataset → cannot do anything useful.
        if live_fp is None:
            logger.warning(
                "Students directory missing or empty: %s", self.students_dir
            )
            self._names.clear()
            self._encodings.clear()
            self._fingerprint = None
            return {
                "status": "no_dataset",
                "students": {},
                "total_encodings": 0,
                "fingerprint": "",
                "skipped": [],
            }

        # Try to load the cache and compare fingerprints.
        if self.encodings_file.exists():
            cached_fp = self._read_cache_fingerprint()
            if cached_fp is not None and cached_fp == live_fp and self.load_encodings():
                logger.info(
                    "Encoding cache hit (fingerprint=%s, %d encodings).",
                    live_fp[:10],
                    len(self._encodings),
                )
                return {
                    "status": "cache_hit",
                    "students": self._student_counts(),
                    "total_encodings": len(self._encodings),
                    "fingerprint": live_fp,
                    "skipped": [],
                }
            logger.info(
                "Encoding cache stale or unreadable — rebuilding "
                "(cached_fp=%s, live_fp=%s).",
                (cached_fp or "")[:10],
                live_fp[:10],
            )
        else:
            logger.info("No encoding cache on disk — building one.")

        summary = self.build_encodings()
        summary["status"] = "cache_rebuilt"
        summary["fingerprint"] = self._fingerprint or ""
        return summary

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
        # Snapshot the dataset state so subsequent startups can short-circuit.
        self._fingerprint = self._compute_dataset_fingerprint()
        self._save_cache()
        logger.info(
            "Encoding build complete – %d encodings for %d students "
            "(fingerprint=%s)",
            len(self._encodings),
            len(summary["students"]),
            (self._fingerprint or "")[:10],
        )
        print("\n" + "=" * 50)
        print("Loaded students:")
        for name in summary["students"].keys():
            print(f"- {name}")
        print(f"Total embeddings count: {len(self._encodings)}")
        print("=" * 50 + "\n")
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
            self._fingerprint = data.get("fingerprint")
            logger.info(
                "Loaded %d encodings from cache (%s, fingerprint=%s)",
                len(self._encodings),
                self.encodings_file,
                (self._fingerprint or "")[:10],
            )
            return True
        except (pickle.UnpicklingError, KeyError, EOFError, ModuleNotFoundError) as exc:
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
        """Load an image, validate quality, and return its first face encoding, or ``None``.

        Validation checks:
            1. Image is not blurry (Laplacian variance > threshold).
            2. Exactly ONE face is detected.
            3. Face resolution is large enough.

        Uses ``num_jitters=settings.NUM_JITTERS`` to re‑sample the face multiple times and
        average the result, producing more accurate encodings.
        """
        # Use cv2 to read → convert to RGB (face_recognition expects RGB)
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            logger.error("Could not read image: %s", image_path)
            return None

        # 1. Blur detection
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance < settings.MIN_FACE_SHARPNESS:
            logger.warning("Skipped %s: Image too blurry (variance: %.2f)", image_path.name, variance)
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Detect faces first to validate
        locations = fr_lib.face_locations(img_rgb, model=self.model)
        
        # 2. Exactly one face
        if len(locations) == 0:
            logger.warning("Skipped %s: No faces found", image_path.name)
            return None
        if len(locations) > 1:
            logger.warning("Skipped %s: Multiple faces found (%d)", image_path.name, len(locations))
            return None

        # 3. Face size validation
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

    def _save_cache(self) -> None:
        """Persist current in-memory encodings + fingerprint to disk **atomically**.

        Strategy: write to ``<file>.tmp-<pid>`` in the same directory,
        ``fsync``, then ``os.replace`` over the real path.  ``os.replace``
        is atomic on both POSIX and Windows, so the cache file is
        either the previous valid version or the new one — *never*
        absent or half-written.  This protects against:

            • The Python process being killed mid-write (Ctrl+C,
              ``taskkill /F``, OS reboot).
            • A registration approval that aborts halfway.
            • Any future caller that crashes after rebuilding.
        """
        target = self.encodings_file
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "names": self._names,
            "encodings": self._encodings,
            "fingerprint": self._fingerprint,
        }

        # NamedTemporaryFile in the SAME directory so os.replace() works
        # across the rename atomically (cross-volume rename is not).
        fd, tmp_path = tempfile.mkstemp(
            prefix=target.stem + ".",
            suffix=".tmp",
            dir=str(target.parent),
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    # Some filesystems / Windows quirks don't allow fsync
                    # on the file descriptor; tolerate it.
                    pass
            os.replace(tmp_path, target)
        except Exception:
            # If anything went wrong, drop the tmp file so it doesn't
            # accumulate.  The previous valid cache is still in place.
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info(
            "Saved %d encodings atomically to %s (fingerprint=%s)",
            len(self._encodings),
            target,
            (self._fingerprint or "")[:10],
        )

    # ── Fingerprint helpers ──────────────────────────────────────────

    def _compute_dataset_fingerprint(self) -> Optional[str]:
        """Return a stable hash that changes iff the image dataset changes.

        Uses the relative path, modification time and size of every
        student image.  Cheap (O(N) ``stat`` calls — milliseconds for
        thousands of files) and immune to false negatives because any
        re-encoded photo bumps mtime and/or size.
        """
        if not self.students_dir.exists():
            return None

        records: List[str] = []
        for student_dir in sorted(self.students_dir.iterdir()):
            if not student_dir.is_dir():
                continue
            for img_path in sorted(student_dir.iterdir()):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in self.image_extensions:
                    continue
                try:
                    st = img_path.stat()
                except OSError:
                    continue
                rel = img_path.relative_to(self.students_dir).as_posix()
                records.append(f"{rel}|{int(st.st_mtime)}|{st.st_size}")

        if not records:
            return None

        digest = hashlib.sha1("\n".join(records).encode("utf-8")).hexdigest()
        return digest

    def _read_cache_fingerprint(self) -> Optional[str]:
        """Cheap probe: pull just the fingerprint out of the .pkl, no full load."""
        try:
            with open(self.encodings_file, "rb") as fh:
                data = pickle.load(fh)
            fp = data.get("fingerprint")
            return str(fp) if fp else None
        except (pickle.UnpicklingError, EOFError, OSError, ModuleNotFoundError):
            return None

    def _student_counts(self) -> Dict[str, int]:
        """Histogram of encodings-per-student from in-memory state."""
        counts: Dict[str, int] = {}
        for name in self._names:
            counts[name] = counts.get(name, 0) + 1
        return counts

