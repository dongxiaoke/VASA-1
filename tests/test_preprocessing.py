import os
import unittest
from training.preprocess import load_data, preprocess_data, save_preprocessed_data

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.data_dir = 'data/raw'
        self.output_dir = 'data/processed'
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Create dummy data
        with open(os.path.join(self.data_dir, 'dummy.wav'), 'w') as f:
            f.write('dummy audio content')
        with open(os.path.join(self.data_dir, 'dummy.jpg'), 'w') as f:
            f.write('dummy image content')

    def test_load_data(self):
        audio_files, image_files = load_data(self.data_dir)
        self.assertGreater(len(audio_files), 0, "No audio files found")
        self.assertGreater(len(image_files), 0, "No image files found")

    def test_preprocess_data(self):
        audio_files, image_files = load_data(self.data_dir)
        audio_data, image_data = preprocess_data(audio_files, image_files)
        self.assertEqual(len(audio_data), len(audio_files), "Mismatch in audio data count")
        self.assertEqual(len(image_data), len(image_files), "Mismatch in image data count")

    def test_save_preprocessed_data(self):
        audio_files, image_files = load_data(self.data_dir)
        audio_data, image_data = preprocess_data(audio_files, image_files)
        save_preprocessed_data(audio_data, image_data, self.output_dir)
        self.assertTrue(os.path.exists(self.output_dir), "Output directory does not exist")
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'audio')), "Audio directory does not exist")
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'image')), "Image directory does not exist")

    def tearDown(self):
        # Clean up dummy data
        os.remove(os.path.join(self.data_dir, 'dummy.wav'))
        os.remove(os.path.join(self.data_dir, 'dummy.jpg'))
        os.rmdir(self.data_dir)
        for root, dirs, files in os.walk(self.output_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.output_dir)

if __name__ == '__main__':
    unittest.main()
