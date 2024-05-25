import unittest
from training.train_latent_space import train_latent_space
from training.train_diffusion import train_diffusion

class TestTraining(unittest.TestCase):

    def test_train_latent_space(self):
        # You might need to set up some dummy data and configuration for the training functions to run
        result = train_latent_space()
        self.assertIsNotNone(result, "Latent space training failed")

    def test_train_diffusion(self):
        # You might need to set up some dummy data and configuration for the training functions to run
        result = train_diffusion()
        self.assertIsNotNone(result, "Diffusion training failed")

if __name__ == '__main__':
    unittest.main()
