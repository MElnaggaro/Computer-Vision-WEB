# vision services package

from app.services.vision.face_detection import FaceDetector
from app.services.vision.face_recognition import FaceRecognizer
from app.services.vision.encoding_manager import EncodingManager
from app.services.vision.attendance_service import AttendanceService

__all__ = [
    "FaceDetector",
    "FaceRecognizer",
    "EncodingManager",
    "AttendanceService",
]
