import pytest
from unittest.mock import MagicMock
import numpy as np

from app.services.vision.emotion_tracker import EmotionTracker
from app.services.vision.attendance_service import AttendanceService

def test_emotion_module_test():
    """1) emotion module test"""
    tracker = EmotionTracker(emotion_interval=1, buffer_size=5, min_stable_samples=3)
    tracker.detector.predict = MagicMock(return_value={"label": "Happy", "confidence": 0.9})
    
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 1 sample
    tracker.update(1, crop, 1)
    assert not tracker.is_stable(1)
    
    # 2 samples
    tracker.update(1, crop, 2)
    assert not tracker.is_stable(1)
    
    # 3 samples -> stable
    tracker.update(1, crop, 3)
    assert tracker.is_stable(1)
    assert tracker.get_smoothed(1)["label"] == "Happy"

def test_known_student_emotion_flow():
    """2) known student emotion flow"""
    att = AttendanceService()
    att.reset_session()
    
    # mark attendance -> no emotion yet
    att.mark_attendance("Test_User", True, 0.99)
    state = att.get_student_state("Test_User")
    assert state["emotion"] == "Analyzing..."
    assert not state["emotion_logged"]
    
    # log emotion
    event = att.log_emotion("Test_User", "Happy", 5)
    assert event is not None
    assert event["event"] == "emotion"
    assert event["mood"] == "Happy"
    
    state = att.get_student_state("Test_User")
    assert state["emotion"] == "Happy"
    assert state["emotion_logged"]
    
    # second time shouldn't log
    event2 = att.log_emotion("Test_User", "Sad", 6)
    assert event2 is None

def test_guest_emotion_flow():
    """3) guest emotion flow"""
    att = AttendanceService()
    att.reset_session()
    
    record = att.register_guest()
    guest_name = record["student"]
    
    state = att.get_student_state(guest_name)
    assert state["emotion"] == "Analyzing..."
    
    event = att.log_emotion(guest_name, "Neutral", 5)
    assert event["mood"] == "Neutral"
    assert event["registered"] is False
    
    state2 = att.get_student_state(guest_name)
    assert state2["emotion"] == "Neutral"

def test_multiple_students_emotion_isolation():
    """4) multiple students emotion isolation"""
    tracker = EmotionTracker(emotion_interval=1, buffer_size=5, min_stable_samples=3)
    
    tracker.detector.predict = MagicMock(side_effect=[
        {"label": "Happy", "confidence": 0.9}, # tid 1
        {"label": "Sad", "confidence": 0.8},   # tid 2
        {"label": "Happy", "confidence": 0.9}, # tid 1
        {"label": "Sad", "confidence": 0.8},   # tid 2
        {"label": "Happy", "confidence": 0.9}, # tid 1
        {"label": "Sad", "confidence": 0.8},   # tid 2
    ])
    
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    
    tracker.update(1, crop, 1)
    tracker.update(2, crop, 1)
    tracker.update(1, crop, 2)
    tracker.update(2, crop, 2)
    
    assert not tracker.is_stable(1)
    assert not tracker.is_stable(2)
    
    tracker.update(1, crop, 3)
    tracker.update(2, crop, 3)
    
    assert tracker.is_stable(1)
    assert tracker.get_smoothed(1)["label"] == "Happy"
    
    assert tracker.is_stable(2)
    assert tracker.get_smoothed(2)["label"] == "Sad"

def test_frontend_update_integration():
    """5) frontend update integration"""
    # Just checking that the logs look right for the frontend to consume
    att = AttendanceService()
    att.reset_session()
    
    att.mark_attendance("Integration_User", True, 0.99)
    att.log_emotion("Integration_User", "Happy", 5)
    
    events = att.records
    assert len(events) == 2
    assert events[0]["event"] == "attendance"
    assert "emotion" not in events[0]
    assert events[1]["event"] == "emotion"
    assert events[1]["mood"] == "Happy"
