"""Unit tests for the EmotionDetection package."""

import unittest
from unittest.mock import Mock, patch

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Verify response formatting and HTTP 400 handling."""

    @patch("emotion_detection.requests.post")
    def test_joy(self, post):
        post.return_value = Mock(
            status_code=200,
            text='{"emotionPredictions":[{"emotion":{"anger":0.01,'
            '"disgust":0.02,"fear":0.03,"joy":0.90,"sadness":0.04}}]}',
        )
        result = emotion_detector("I am glad this happened")
        self.assertEqual(result["dominant_emotion"], "joy")
        self.assertEqual(
            set(result),
            {"anger", "disgust", "fear", "joy", "sadness", "dominant_emotion"},
        )

    @patch("emotion_detection.requests.post")
    def test_anger(self, post):
        post.return_value = Mock(
            status_code=200,
            text='{"emotionPredictions":[{"emotion":{"anger":0.91,'
            '"disgust":0.02,"fear":0.03,"joy":0.01,"sadness":0.03}}]}',
        )
        result = emotion_detector("I am really mad about this")
        self.assertEqual(result["dominant_emotion"], "anger")

    @patch("emotion_detection.requests.post")
    def test_bad_request(self, post):
        post.return_value = Mock(status_code=400, text="")
        result = emotion_detector("")
        self.assertIsNone(result["dominant_emotion"])


if __name__ == "__main__":
    unittest.main()
