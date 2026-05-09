# vision services package

from app.services.vision.face_detection import FaceDetector
from app.services.vision.face_recognizer import FaceRecognizer
from app.services.vision.encoding_manager import EncodingManager
from app.services.vision.attendance_service import AttendanceService
from app.services.vision.webcam_runner import ClassroomCamera

__all__ = [
    "FaceDetector",
    "FaceRecognizer",
    "EncodingManager",
    "AttendanceService",
    "ClassroomCamera",
]
