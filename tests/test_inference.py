import unittest
from inference.generate_video import generate_video

class TestInference(unittest.TestCase):

    def test_generate_video(self):
        # You might need to set up some dummy data and configuration for the inference functions to run
        result = generate_video()
        self.assertTrue(result, "Video generation failed")

if __name__ == '__main__':
    unittest.main()
