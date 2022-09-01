from __future__ import annotations

from blpytorchlightning.dataset_components.AIMLoader import AIMLoader


class AIMLoaderRAM(AIMLoader):
    """ Class for loading AIM images (and masks) from a directory, aligning them, and returning as numpy arrays.
    Does not load images from file on-demand, but rather loads the whole dataset into RAM up-front. """
    _image_dict: Dict[Dict[np.ndarray]]

    def __init__(self, *args, **kwargs) -> None:
        """ Initialization method """
        super().__init__(*args, **kwargs)
        self._load_images_into_memory()

    def _load_images_into_memory(self) -> None:
        """ Iterate through the image list and load everything into memory."""
        self._image_dict = {}
        for image_fn in self._image_list:
            self._image_dict[image_fn] = {}
            image, masks = self._get_image_and_masks(image_fn)
            self._image_dict[image_fn]['image'] = image
            self._image_dict[image_fn]['masks'] = masks

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the image and mask for a given image name (from RAM).

        Parameters
        ----------
        image_fn : str
            The full filepath to the image on disk (and the key for the dict containing the data)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            First element: the density image as a numpy array of shape (1,D,H,W)
            Second element: the masks as a numpy array of shape (3,D,H,W).
            Mask order is cortical, trabecular, background
        """
        image_fn = self._image_list[idx]
        return self._get_image_and_mask_from_memory(image_fn)

    def _get_image_and_mask_from_memory(self, image_fn) -> Tuple[np.ndarray, np.ndarray]:
        """ Get the image and mask from memory

        Parameters
        ----------
        image_fn : str
            The full filepath to the image on disk (and the key for the dict containing the data)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            First element: the density image as a numpy array of shape (1,D,H,W)
            Second element: the masks as a numpy array of shape (3,D,H,W).
            Mask order is cortical, trabecular, background
        """
        image = self._image_dict[image_fn]['image']
        masks = self._image_dict[image_fn]['masks']
        return image, masks
