"""Environment with some preprocessing"""

import numpy as np
from scipy import ndimage

import gym
from gym import wrappers


class NonRGBImage(Exception):
    """Raised when a RGB image is expected but not provided"""

    pass


class ProcessedEnvironnement:
    """Wrapper around a OpenAI environment to add some pre-processing"""

    def __init__(self, env_id, outdir=None, wrappers_cond=False,
                 grayscale=True, new_shape=(84, 84)):
        """Init

        :param env_id: id of the OpenAI environment
        :param outdir: output directory
        :param wrappers_cond: boolean, whether or not to call wrappers.Monitor
        :param grayscale: boolean, whether or not to turn the input into
         a grayscale image
        :param new_shape: tuple, new shape of the resized input
        """

        self.env = gym.make(env_id)
        if wrappers_cond:
            self.env = wrappers.Monitor(self.env, directory=outdir, force=True)

        self.action_space = self.env.action_space

        self.grayscale = grayscale
        self.new_shape = new_shape

    @staticmethod
    def _gaussian_normalization(image):
        """Gaussian normalization of the input

        :param image: input image
        :return: image centered and with standard deviation of 1
        """

        image_processed = image - np.mean(image)
        std = np.std(image_processed)
        std = std if std > 0 else 1
        image_processed /= std

        return image_processed

    def _preprocess(self, ob):
        """Preprocess input observation

        :param ob: input observation, typically an nd.array (nx, ny, 3)
        :return: a preprocessed observation
        """

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

        #ob_processed = self._gaussian_normalization(ob_processed)

        return ob_processed

    @staticmethod
    def _reshape(image, new_shape):
        """Reshape the image to new_shape

        :param image: input np.array
        :param new_shape: tuple, new shape of the image
        :return: reshaped image
        """

        input_shape = np.array(image.shape).astype(np.float)
        output_shape = np.array(new_shape + (1,)).astype(np.float)
        zoom_factors = output_shape / input_shape
        image_reshaped = ndimage.zoom(image, zoom_factors, order=1)

        return image_reshaped

    @staticmethod
    def _turn_rgb_grayscale(image):
        """Turn RGB image to grayscale

        :param image: input image with shape (nx, ny, 3)
        :return: grayscale image with shape (nx, ny, 1)
        """

        image_grayscale = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        return image_grayscale[..., None]

    def close(self):
        """Close the environment"""

        self.env.close()

    def reset(self):
        """Reset the environment

        :return: initial preprocessed observation
        """

        ob = self.env.reset()

        ob_preprocessed = self._preprocess(ob)

        return ob_preprocessed

    def seed(self, int):
        """Set the random seed of the environment"""

        self.env.seed(int)

    def step(self, action):
        """Take a step in the environment

        :param action: int, action id
        :return: tuple with observed preprocessed observation, reward, boolean
         whether the episode is done or not, info
        """

        ob, reward, done, info = self.env.step(action)

        ob_preprocessed = self._preprocess(ob)
        if done:
            reward = -1
        if reward == 0:
            reward = 0.1
        reward = np.clip(reward, -1, 1)

        return ob_preprocessed, reward, done, info
