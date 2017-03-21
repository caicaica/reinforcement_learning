"""Environment with some preprocessing"""

import numpy as np
from scipy import ndimage

import gym
from gym import wrappers


class NonRGBImage(Exception):
    """Raised when a RGB image is expected but not provided"""

    pass


class ProcessedEnvironnement:

    def __init__(self, env_id, outdir=None, wrappers_cond=False,
                 grayscale=True, new_shape=(128, 128)):

        self.env = gym.make(env_id)
        if wrappers_cond:
            self.env = wrappers.Monitor(self.env, directory=outdir, force=True)

        self.action_space = self.env.action_space

        self.grayscale = grayscale
        self.new_shape = new_shape

    @staticmethod
    def _gaussian_normalization(image):

        image_processed = image - np.mean(image)
        sigma = np.std(image_processed)
        sigma = sigma if sigma > 0 else 1
        image_processed /= sigma

        return image_processed

    def _preprocess(self, ob):

        ob_processed = ob.copy()

        if self.grayscale:
            if ob.ndim == 3 and ob.shape[-1] == 3:
                ob_processed = self._turn_rgb_grayscale(ob_processed)
            else:
                raise NonRGBImage(
                    'Unexpected image shape: {}'.format(ob.shape)
                )
        if self.new_shape != ob.shape:
            ob_processed = self._reshape(ob_processed, self.new_shape)

        ob_processed = self._gaussian_normalization(ob_processed)

        return ob_processed

    @staticmethod
    def _reshape(image, new_shape):

        zoom_factors = np.array(new_shape + (1,)).astype(np.float) / np.array(image.shape).astype(np.float)
        image_reshaped = ndimage.zoom(image, zoom_factors, order=1)

        return image_reshaped

    @staticmethod
    def _turn_rgb_grayscale(image):

        image_grayscale = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        return image_grayscale[..., None]

    def close(self):

        self.env.close()

    def reset(self):

        ob = self.env.reset()

        ob_preprocessed = self._preprocess(ob)

        return ob_preprocessed

    def seed(self, int):

        self.env.seed(int)

    def step(self, action):

        ob, reward, done, info = self.env.step(action)

        ob_preprocessed = self._preprocess(ob)

        return ob_preprocessed, reward, done, info
