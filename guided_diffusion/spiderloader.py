import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from skimage.transform import resize 
from skimage import img_as_ubyte
import SimpleITK as sitk 

def normalize_image(image_slice):
    """
    Normalize image slice to the range [0, 1].
    """
    image_slice = image_slice.astype(np.float32)
    min_val, max_val = np.min(image_slice), np.max(image_slice)

    if max_val > min_val:  # Avoid division by zero
        image_slice = (image_slice - min_val) / (max_val - min_val)
    else:
        image_slice = np.zeros_like(image_slice)

    return image_slice

def preprocess_spider_slices(image_array, mask_array, image_size_tuple):
    """
    Preprocesses slices: selects middle 5, resizes to image_size_tuple, normalizes.
    """
    processed_image_slices = []
    processed_mask_slices = []

    mid_idx = image_array.shape[0] // 2
    start_idx = max(0, mid_idx - 2)
    end_idx = min(image_array.shape[0], mid_idx + 3) # Get 5 slices: mid-2, mid-1, mid, mid+1, mid+2 this is because its 3d volume mri and usually the middle slices are the ones with most details

    for i in range(start_idx, end_idx):
        # Use image_size_tuple for resizing
        image_slice = resize(image_array[i, :, :], image_size_tuple, mode="reflect", order=3, anti_aliasing=True)
        mask_slice = resize(mask_array[i, :, :], image_size_tuple, mode="reflect", order=0, preserve_range=True, anti_aliasing=False)

        image_slice = normalize_image(image_slice) 
        image_slice = img_as_ubyte(image_slice) 
        mask_slice = mask_slice.astype(np.uint8)

        processed_image_slices.append(image_slice)
        processed_mask_slices.append(mask_slice)
    return processed_image_slices, processed_mask_slices

class SPIDERDataset(Dataset):
    def __init__(self, data_path, image_size, transform=None, mode='Training'):
        """
        Args:
            data_path (str): Base directory.
            image_size (int): Target size (e.g., 256). Assumes square images.
            transform (callable, optional): Optional AUGMENTATIONS to be applied.
            mode (str): Training/Validation/Test.
        """
        super().__init__()
        self.data_path = data_path
        self.image_size_tuple = (image_size, image_size) 
        self.transform = transform 
        self.mode = mode

        self.image_slices = []
        self.mask_slices = []
        self.slice_info = [] # Store info like (original_filename, slice_index_in_volume)

        image_dir = os.path.join(self.data_path, "images")
        mask_dir = os.path.join(self.data_path, "masks")

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.isdir(mask_dir):
             raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".mha")])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".mha")])

        print(f"Found {len(image_files)} image files and {len(mask_files)} mask files.")

        if len(image_files) != len(mask_files):
            print("Warning: Number of image files and mask files do not match.")

        processed_volumes = 0
        for img_file, mask_file in zip(image_files, mask_files):
            # We only want t2 and not SPACE
            if "t2" not in img_file or "SPACE" in img_file:
                continue


            img_base = img_file.replace('.mha', '')
            mask_base = mask_file.replace('.mha', '').replace('_mask', '') # Adjust if needed
            if img_base != mask_base:
                 print(f"Warning: Skipping potentially mismatched pair: {img_file}, {mask_file}")
                 continue


            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            try:
                image_itk = sitk.ReadImage(img_path)
                mask_itk = sitk.ReadImage(mask_path)

                image_array = sitk.GetArrayFromImage(image_itk) # Shape: (z, y, x)
                mask_array = sitk.GetArrayFromImage(mask_itk)   # Shape: (z, y, x)

                # Orientation Check and Correction
                direction = image_itk.GetDirection()
                # Direction matrix flattened in column-major order: (d11, d21, d31, d12, d22, d32, d13, d23, d33)
                # Third column (axis 2, typically Z in ITK): (d13, d23, d33) -> indices (6, 7, 8)
                third_column = (direction[6], direction[7], direction[8])

                # Check if Z-axis direction cosine is close to (0, 0, 1) which suggests (x, y, z) order in the array
                if np.allclose(third_column, (0, 0, 1), atol=1e-3):
                    # Transpose (x, y, z) -> (z, y, x)
                    image_array = np.transpose(image_array, (2, 1, 0))
                    mask_array = np.transpose(mask_array, (2, 1, 0))

                    # Rotate each slice 90 degrees counter-clockwise
                    image_array = np.rot90(image_array, k=1, axes=(1, 2))
                    mask_array = np.rot90(mask_array, k=1, axes=(1, 2))

                
                # Returns lists of 2D slices
                processed_img_slices, processed_mask_slices = preprocess_spider_slices(
                    image_array, mask_array, self.image_size_tuple
                )

                # Store individual slices
                num_slices_in_volume = len(processed_img_slices)
                mid_idx = image_array.shape[0] // 2
                start_idx = max(0, mid_idx - 2)

                for i in range(num_slices_in_volume):
                    self.image_slices.append(processed_img_slices[i]) 
                    self.mask_slices.append(processed_mask_slices[i])  
                    
                    slice_original_index = start_idx + i
                    self.slice_info.append({'filename': img_file, 'slice_index': slice_original_index})

                processed_volumes += 1

            except Exception as e:
                print(f"Error processing file pair: {img_file}, {mask_file}. Error: {e}")
                

        print(f"Finished preprocessing. Processed {processed_volumes} volumes, loaded {len(self.image_slices)} slices.")
        if len(self.image_slices) == 0:
            print("Warning: No slices were loaded. Check file paths, filtering criteria, and potential errors.")


    def __len__(self):
        """Return the total number of individual slices."""
        return len(self.image_slices)

    def __getitem__(self, index):
        img_slice_np = self.image_slices[index]
        mask_slice_np = self.mask_slices[index].astype(np.int64)
        slice_info = self.slice_info[index]

        img_pil = Image.fromarray(img_slice_np).convert('RGB')
        mask_pil = Image.fromarray(mask_slice_np.astype(np.uint8)).convert('L')

        if self.transform:
             state = torch.get_rng_state()
             img_pil = self.transform(img_pil)
             torch.set_rng_state(state)
             mask_pil = self.transform(mask_pil)

        img_tensor = transforms.ToTensor()(img_pil)

        mask_np_final = np.array(mask_pil).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_np_final).long()
        mask_tensor = mask_tensor.unsqueeze(0)

        return (img_tensor, mask_tensor, slice_info['filename'] + f"_slice{slice_info['slice_index']}")



