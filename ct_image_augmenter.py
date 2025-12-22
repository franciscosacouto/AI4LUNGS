import albumentations as A
import cv2
import numpy
import random
import torch
import torch.nn.functional as F

def patch_shuffle(img, patch_size=2):
    if isinstance(img, numpy.ndarray):
        img = torch.tensor(img)

    if img.dim() != 3:
        raise ValueError("Expected 3D tensor with shape (H, W, C)")

    H, W, C = img.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"Image dimensions ({H}, {W}) must be divisible by patch size {patch_size}")

    # Reshape to patches: (H//p, W//p, p, p, C)
    patches = img.view(H // patch_size, patch_size, W // patch_size, patch_size, C)
    patches = patches.permute(0, 2, 1, 3, 4)  # (n_patches_h, n_patches_w, patch_size, patch_size, C)

    # Shuffle pixels inside each patch
    for i in range(patches.shape[0],):
        for j in range(patches.shape[1]):
            for c in range(C):
                patch = patches[i, j, :, :, c]
                flat = patch.flatten()
                shuffled = flat[torch.randperm(flat.numel())]
                patches[i, j, :, :, c] = shuffled.view(patch_size, patch_size)

    # Reconstruct the image
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()
    shuffled_img = patches.view(H, W, C)

    return shuffled_img


class PatchShuffleTransform(A.ImageOnlyTransform):
    def __init__(self, patch_size=16, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.patch_size = patch_size

    def apply(self, img, **params):
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        img_tensor = torch.tensor(img)
        shuffled_img = patch_shuffle(img_tensor, patch_size=self.patch_size)
        return shuffled_img.numpy()

    def get_transform_init_args_names(self):
        return ("patch_size",)


class CTImageAugmenter:
    """
    A class for applying random 2D data augmentations to CT images and masks.

    This class implements random 2D data augmentations based on the BigAug method proposed by Ling 
    Zhang et al. (2020) in the paper "Generalizing Deep Learning for Medical Image Segmentation 
    to Unseen Domains via Deep Stacked Transformation." BigAug consists of a sequence of nine 
    deep stacked transformations designed to enhance the generalization of 3D medical image 
    segmentation models to unseen domains. The transformations are categorized into three groups:

    - **Image Quality**: Includes noise insertion, blur insertion, and sharpness adjustment to 
      simulate variations in image quality.
    - **Image Appearance**: Involves brightness, contrast, and intensity perturbations to emulate 
      the characteristics of different scanners and imaging protocols.
    - **Spatial Configuration**: Comprises rotation, scaling, and deformation to account for intra-
      and inter-patient differences and variations in imaging quality.

    This class differs from the original BigAug framework by omitting sharpness adjustment and 
    intensity perturbation while incorporating vertical and horizontal flips, as well as 
    translation and shearing. Therefore, the transformations covered by this class include 
    rotation, translation, shearing, flipping, elastic deformation, blurring, brightness/contrast 
    adjustments, and noise insertion. The number of augmentations applied per call is determined 
    by the configuration.

    Attributes:
        number_of_augmentations (int): The number of random augmentations to apply.
        data_augmentations (list): A list of augmentation transformations to randomly choose from.
    """
    def __init__(self, parameters):
        self.parameters = parameters

        # Basic augmentations (Albumentations-compatible)
        self.basic_geometric = [
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.ShiftScaleRotate( # values tested on NLST dataset
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                p=1.0
            ),
            A.Affine(shear=(-5, 5), p=1.0)
        ]

        self.basic_occlusion = [
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.7)
        ]

        self.basic_intensity_ops = [
            A.RandomGamma(gamma_limit=(50, 150), p=0.7),
            A.CLAHE(clip_limit=1.1, p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.3, 0.3),
                p=0.7
            )
        ]

        self.basic_noise = [
            A.GaussNoise(std_range= (0.1, 0.15), mean = (0.0, 0.0), p=1.0), # Introduces too much noise
            #A.ISONoise(p=0.7),
            A.SaltAndPepper(amount= (0.005, 0.005), p=0.7),
        ]

        self.basic_filtering = [
            A.MotionBlur(blur_limit=5, p=0.7),
            A.MedianBlur(blur_limit=5, p=0.7),
            A.GaussianBlur(blur_limit=7, p=0.7),
            PatchShuffleTransform(patch_size=4, p=0.7)
        ]

        # Deformable augmentations
        self.deformable = [
            A.ElasticTransform(
                alpha=50.0,
                sigma=20.0,
                p=0.7
            ),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.0, p=0.7)
        ]


    def __call__(self, image, mask=None):
        """
        Applies random augmentations to the CT image, and optionally to the mask.

        Args:
            image (torch.Tensor): The CT image tensor to augment, in shape (C, H, W).
            mask (torch.Tensor, optional): Optional mask tensor to augment alongside the image, 
                in shape (C, H, W). Defaults to None if no mask is provided.

        Returns:
            torch.Tensor or tuple: If `mask` is not provided, returns the augmented image tensor.
            If `mask` is provided, returns a tuple containing the augmented image and mask tensors.

        Legend:
            H = Height of the image
            W = Width of the image
            C = Number of channels in the image
        """
        if image.ndim == 2:  # Check if the numpy array image shape is (H, W)
            image = image.reshape(1, *image.shape)  # Convert numpy array image from (H, W) to (1, H, W).
        image = numpy.transpose(image, axes=(1, 2, 0))
        #image_np = image.transpose(1, 2, 0)  # to HWC
        #print(image.shape)
        #print(image.dtype)

        # Choose 3 random augmentations from the basic set
        basic_aug_1 = random.choice(
            self.basic_geometric)
        #print(f"Applying basic augmentation: {basic_aug_1.__class__.__name__}")
        basic_aug_2 = random.choice(
            self.basic_occlusion +
            self.basic_intensity_ops)
        #print(f"Applying basic augmentation: {basic_aug_2.__class__.__name__}")
        basic_aug_3 = random.choice(
            self.basic_noise +
            self.basic_filtering
        )
        # Apply basic augmentation
        #print(f"Applying basic augmentation: {basic_aug_3.__class__.__name__}")
        #if isinstance(basic_aug, A.BasicTransform):
        image_np = basic_aug_1(image=image)['image']
        image_np = basic_aug_2(image=image_np)['image']
        image_np = basic_aug_3(image=image_np)['image']

        deform_aug = random.choice(self.deformable)
        #print(f"Applying deformable augmentation: {deform_aug.__class__.__name__}")
        image_np = deform_aug(image=image_np)['image']
        #print(f"Image shape after augmentations: {image_np.shape}")
        #print(f"Image dtype after augmentations: {image_np.dtype}")
        # Convert back to tensor and permute to C, H, W
        image_tensor = torch.tensor(image_np.transpose(2, 0, 1))

        #image_tensor = torch.tensor(image_np.transpose(2, 0, 1))

        return image_np if mask is None else (image_tensor, mask)


class CTImageAugmenter3D:
    """
    A class to apply consistent 2D augmentations slice-wise to 3D CT images and masks using ReplayCompose.

    This version enforces that one augmentation from each category is selected, and applied identically to all slices.
    """

    def __init__(self, parameters):
        self.parameters = parameters

        # --- Define augmentation categories ---
        # Basic augmentations (Albumentations-compatible)
        self.basic_geometric = [
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.ShiftScaleRotate( # values tested on NLST dataset
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                p=1.0
            ),
            A.Affine(shear=(-5, 5), p=1.0)
        ]

        self.basic_occlusion = [
            A.CoarseDropout3D(
                num_holes_range=(3, 6),
                hole_depth_range=(0.1, 0.2),
                hole_height_range=(0.1, 0.2),
                hole_width_range=(0.1, 0.2),
                fill=0,  # Updated key name for latest Albumentations
                p=0.7
            )
        ]

        self.basic_intensity_ops = [
            A.RandomGamma(gamma_limit=(50, 150), p=0.7),
            A.CLAHE(clip_limit=1.1, p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.3, 0.3),
                p=0.7
            )
        ]

        self.basic_noise = [
            A.GaussNoise(std_range= (0.1, 0.15), mean = (0.0, 0.0), p=1.0), # Introduces too much noise
            #A.ISONoise(p=0.7),
            A.SaltAndPepper(amount= (0.005, 0.005), p=0.7),
        ]

        self.basic_filtering = [
            A.MotionBlur(blur_limit=5, p=0.7),
            A.MedianBlur(blur_limit=5, p=0.7),
            A.GaussianBlur(blur_limit=7, p=0.7),
        ]

        self.deformable = [
            A.ElasticTransform(
                alpha=50.0,
                sigma=20.0,
                p=0.7
            ),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.0, p=0.7)
        ]


    def __call__(self, volume, mask=None):
        """
        Apply consistent random 2D augmentations to all slices in a 3D volume (and mask, if provided).

        Args:
            volume (np.ndarray): 3D numpy array of shape (Z, H, W)
            mask (np.ndarray): Optional mask of shape (Z, H, W)

        Returns:
            Augmented volume (and mask if provided)
        """
        assert volume.ndim == 3, "Volume must have shape (Z, H, W)"
        Z, H, W = volume.shape
        print(f"Volume shape: {volume.shape}")

        if mask is not None:
            assert mask.shape == volume.shape, "Mask must have the same shape as volume"

        # Convert to ZHWC
        if volume.ndim == 3:
            volume = numpy.expand_dims(volume, axis=-1)
            print(volume.shape)
        if mask is not None and mask.ndim == 3:
            mask = numpy.expand_dims(mask, axis=-1)

        # Randomly pick one transformation from each group
        chosen_transforms = [
            random.choice(self.basic_geometric),
            random.choice(self.basic_occlusion + self.basic_intensity_ops),
            random.choice(self.basic_noise + self.basic_filtering),
            random.choice(self.deformable)
        ]

        coarse_dropout_transform = None
        for t in chosen_transforms:
            if isinstance(t, A.CoarseDropout3D):
                print(f"Using coarse dropout transform: {t}")
                coarse_dropout_transform = t
                chosen_transforms.remove(t)
                break

        composed = A.ReplayCompose(chosen_transforms)

        # Apply to first slice and get replay
        first_input = {"image": volume[0]}
        # Turn into RGB if necessary
        if volume[0].ndim == 2:
            first_input["image"] = numpy.stack([first_input["image"]] * 3, axis=-1)
        if mask is not None:
            first_input["mask"] = mask[0]

        first_result = composed(**first_input)
        replay = first_result["replay"]
        if first_result['image'].ndim == 3 and first_result['image'].shape[-1] == 3:
                first_result['image'] = numpy.mean(first_result['image'], axis=-1) # Uses mean to compute grayscale

        transformed_volume = [first_result["image"]]
        transformed_mask = [first_result["mask"]] if mask is not None else None

        # Apply same transform to remaining slices
        for i in range(1, volume.shape[0]):
            input_data = {"image": volume[i]}
            # Turn into RGB if necessary
            if volume[i].ndim == 2:
                input_data["image"] = numpy.stack([input_data["image"]] * 3, axis=-1)
            if mask is not None:
                input_data["mask"] = mask[i]

            result = A.ReplayCompose.replay(replay, **input_data)
            # Transform into grayscale if necessary
            if result['image'].ndim == 3 and result['image'].shape[-1] == 3:
                result['image'] = numpy.mean(result['image'], axis=-1) # Uses mean to compute grayscale

            transformed_volume.append(result["image"])

            if mask is not None:
                transformed_mask.append(result["mask"])

        transformed_volume = numpy.stack(transformed_volume, axis=0)
        if mask is not None:
            transformed_mask = numpy.stack(transformed_mask, axis=0)
            return transformed_volume, transformed_mask
        

        if coarse_dropout_transform is not None:
            volume_input = {"volume": transformed_volume}
            if mask is not None:
                volume_input["mask"] = transformed_mask
            result = A.Compose([coarse_dropout_transform])(**volume_input)
            transformed_volume = result["volume"]
            if mask is not None:
                transformed_mask = result["mask"]

        return transformed_volume