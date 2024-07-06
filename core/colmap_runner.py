"""Helper tool to run colmap."""
import os
import subprocess
from typing import Optional


# The resolution of the longest side of the image.
IMAGE_RESOLUTION_LONG_SIDE = 1920


class ColmapProcessor(object):
    """A helper class to run colmap.

    Attributes:
        image_root (str): The root directory containing input images.
        work_space (str): The directory for Colmap workspace.
    """
    def __init__(self) -> None:
        """Initialize the ColmapProcessor.

        Args:
            image_root (str): The root directory containing input images.
            work_space (str): The directory for Colmap workspace.
        """
        self.image_root = ""
        self.work_space = ""

    def run_colmap(self, image_root: str, work_space: str, use_extern_features: bool = False):
        """Runs colmap for the given data.
        
        Args:
            image_root (str): The root directory containing input images.
            work_space (str): The directory for Colmap workspace.
        """
        self.image_root = image_root
        self.work_space = work_space
        if not os.path.exists(self.work_space):
            os.mkdir(self.work_space)

        if not use_extern_features:
            self.feature_extraction()
            self.feature_matching()
        self.camera_pose_estimation()

    def feature_extraction(self, camera_params: Optional[str] = None):
        """Perform feature extraction.

        Args:
            camera_params (str, optional): Camera parameters for known camera
            model. Format: "focal_length,principal_point_x,principal_point_y".
        """
        # Build the path to the database file and the image directory
        database_path = os.path.join(self.work_space, "database.db")
        image_path = self.image_root

        # Construct the command for feature extraction
        command = [
            "colmap", "feature_extractor",
            "--database_path", database_path,
            "--image_path", image_path,
            "--ImageReader.camera_model", "SIMPLE_PINHOLE",
            "--SiftExtraction.max_image_size", str(IMAGE_RESOLUTION_LONG_SIDE)
        ]

        # Append camera parameters if provided
        if camera_params:
            command.extend(["--ImageReader.camera_params", camera_params])

        # Execute the command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during feature extraction: {e}")

    def feature_matching(self):
        """Perform feature matching."""
        # Build the path to the database file
        database_path = os.path.join(self.work_space, "database.db")

        # Construct the command for feature matching
        command = ["colmap", "exhaustive_matcher", "--database_path",
                   database_path]

        # Execute the command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during feature matching: {e}")

    def camera_pose_estimation(self):
        """Perform camera pose estimation."""
        # Build the paths to the database file and the output directory
        database_path = os.path.join(self.work_space, "database.db")
        output_path = os.path.join(self.work_space, "sparse")

        # Construct the command for camera pose estimation
        command = [
            "colmap", "mapper",
            "--database_path", database_path,
            "--image_path", self.image_root,
            "--output_path", output_path,
            "--Mapper.ba_refine_principal_point", "true"
        ]

        # Execute the command
        try:
            os.makedirs(output_path, exist_ok=True)
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during camera pose estimation: {e}")


# Usage example
if __name__ == "__main__":
    image_root = "/path/to/your/images"  # Input images root directory
    work_space = "/path/to/your/workspace"  # Colmap workspace directory

    # Create ColmapProcessor instance and execute steps
    colmap_processor = ColmapProcessor()
    colmap_processor.run_colmap(image_root, work_space)
