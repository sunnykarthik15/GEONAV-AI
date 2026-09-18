"""Classical visual front-end for feature detection, tracking, and relative motion estimation."""

from typing import Optional, Tuple
import cv2
import numpy as np

from geonav.config.settings import CameraConfig, VIOConfig
from geonav.vio.types import VisualTrackingResult


class VisualFrontEnd:
    """Performs classical monocular visual feature detection, optical flow tracking,

    and Essential-matrix-based relative motion estimation.

    Note on Monocular Scale:
        Monocular Essential matrix decomposition recovers translation only up to an
        unknown arbitrary scale (unit direction vector). Metric translation cannot be
        obtained from monocular vision alone.
    """

    def __init__(
        self,
        camera_config: Optional[CameraConfig] = None,
        vio_config: Optional[VIOConfig] = None,
    ) -> None:
        """Initialize the visual front-end with camera and VIO parameters.

        Args:
            camera_config: Camera calibration parameters.
            vio_config: VIO algorithm parameters.
        """
        self.cam_cfg = camera_config or CameraConfig()
        self.vio_cfg = vio_config or VIOConfig()

        self.max_features = self.vio_cfg.max_features
        self.min_features = self.vio_cfg.min_features
        self.ransac_threshold = self.vio_cfg.ransac_threshold
        self.ransac_prob = self.vio_cfg.ransac_prob

        # Set up camera intrinsic matrix K
        fx = self.cam_cfg.fx if self.cam_cfg.fx is not None else float(self.cam_cfg.width)
        fy = self.cam_cfg.fy if self.cam_cfg.fy is not None else float(self.cam_cfg.height)
        cx = self.cam_cfg.cx if self.cam_cfg.cx is not None else (self.cam_cfg.width / 2.0)
        cy = self.cam_cfg.cy if self.cam_cfg.cy is not None else (self.cam_cfg.height / 2.0)

        self.K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        dist = self.cam_cfg.distortion_coeffs
        self.dist_coeffs = np.array(dist, dtype=np.float64) if dist is not None else np.zeros(4, dtype=np.float64)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Validate and convert an incoming camera image to single-channel 8-bit grayscale.

        Args:
            image: Input image array.

        Returns:
            np.ndarray: 2D 8-bit unsigned integer grayscale image.

        Raises:
            ValueError: If the image is empty, has invalid dimensions, or non-finite values.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray image, got {type(image)}")
        if image.size == 0 or len(image.shape) < 2:
            raise ValueError("Input image is empty or has fewer than 2 dimensions")
        if not np.all(np.isfinite(image)):
            raise ValueError("Input image contains non-finite values (NaN or Inf)")

        # Convert to uint8 if necessary
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                # If normalized float in [0, 1], scale to [0, 255]
                if image.max() <= 1.0:
                    image = (image * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    image = image.clip(0, 255).astype(np.uint8)
            else:
                image = image.clip(0, 255).astype(np.uint8)

        # Convert multi-channel to grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif image.shape[2] == 1:
                gray = image[:, :, 0]
            else:
                raise ValueError(f"Unsupported number of image channels: {image.shape[2]}")
        else:
            gray = image

        return gray

    def detect_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Detect classical Shi-Tomasi corner features.

        Args:
            gray_image: 8-bit single-channel grayscale image.

        Returns:
            np.ndarray: Array of detected feature pixel coordinates of shape (N, 2).
        """
        corners = cv2.goodFeaturesToTrack(
            gray_image,
            maxCorners=self.max_features,
            qualityLevel=0.01,
            minDistance=10.0,
            blockSize=3,
        )
        if corners is None or len(corners) == 0:
            return np.empty((0, 2), dtype=np.float32)

        return corners.reshape(-1, 2).astype(np.float32)

    def track_features(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        pts_prev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Track feature points using Lucas-Kanade optical flow with bidirectional consistency check.

        Args:
            prev_gray: Previous 8-bit grayscale image.
            curr_gray: Current 8-bit grayscale image.
            pts_prev: Array of 2D feature coordinates in the previous image (N, 2).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (tracked_points_prev, tracked_points_curr).
        """
        if pts_prev is None or len(pts_prev) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        p0 = pts_prev.reshape(-1, 1, 2).astype(np.float32)

        # Forward tracking: prev -> curr
        p1, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            p0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        if p1 is None or status_fwd is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        # Backward tracking for consistency check: curr -> prev
        p0_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray,
            prev_gray,
            p1,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        if p0_back is None or status_bwd is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        p0_flat = p0.reshape(-1, 2)
        p1_flat = p1.reshape(-1, 2)
        p0_back_flat = p0_back.reshape(-1, 2)

        # Forward-backward position error
        fb_err = np.linalg.norm(p0_flat - p0_back_flat, axis=1)

        h, w = curr_gray.shape[:2]
        in_bounds = (
            (p1_flat[:, 0] >= 0)
            & (p1_flat[:, 0] < w)
            & (p1_flat[:, 1] >= 0)
            & (p1_flat[:, 1] < h)
        )

        valid_mask = (
            (status_fwd.ravel() == 1)
            & (status_bwd.ravel() == 1)
            & (fb_err < 1.0)
            & in_bounds
            & np.all(np.isfinite(p1_flat), axis=1)
        )

        return p0_flat[valid_mask], p1_flat[valid_mask]

    def estimate_relative_motion(
        self,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
    ) -> VisualTrackingResult:
        """Estimate relative camera rotation and unit translation direction via Essential matrix.

        Args:
            pts_prev: Matched feature points in previous frame (N, 2).
            pts_curr: Matched feature points in current frame (N, 2).

        Returns:
            VisualTrackingResult: Result containing relative pose or explicit failure reason.
        """
        if pts_prev is None or pts_curr is None:
            return VisualTrackingResult(
                success=False,
                num_tracked=0,
                num_inliers=0,
                status_message="Point arrays are None",
            )

        if len(pts_prev) != len(pts_curr):
            return VisualTrackingResult(
                success=False,
                num_tracked=0,
                num_inliers=0,
                status_message=f"Mismatched point counts: {len(pts_prev)} vs {len(pts_curr)}",
            )

        num_tracked = len(pts_prev)
        if num_tracked < self.min_features:
            return VisualTrackingResult(
                success=False,
                num_tracked=num_tracked,
                num_inliers=0,
                points_prev=pts_prev,
                points_curr=pts_curr,
                status_message=f"Insufficient tracked features ({num_tracked} < {self.min_features})",
            )

        # Check for non-finite values or degenerate spatial distribution
        if not np.all(np.isfinite(pts_prev)) or not np.all(np.isfinite(pts_curr)):
            return VisualTrackingResult(
                success=False,
                num_tracked=num_tracked,
                num_inliers=0,
                status_message="Non-finite feature coordinates detected",
            )

        # Degeneracy check: points must not be completely collinear or clustered
        std_prev = np.std(pts_prev, axis=0)
        std_curr = np.std(pts_curr, axis=0)
        if std_prev[0] < 1.0 or std_prev[1] < 1.0 or std_curr[0] < 1.0 or std_curr[1] < 1.0:
            return VisualTrackingResult(
                success=False,
                num_tracked=num_tracked,
                num_inliers=0,
                status_message="Degenerate spatial feature distribution (collinear or zero variance)",
            )

        try:
            # Estimate Essential matrix with RANSAC
            E, mask = cv2.findEssentialMat(
                pts_prev,
                pts_curr,
                cameraMatrix=self.K,
                method=cv2.RANSAC,
                prob=self.ransac_prob,
                threshold=self.ransac_threshold,
            )

            if E is None or mask is None:
                return VisualTrackingResult(
                    success=False,
                    num_tracked=num_tracked,
                    num_inliers=0,
                    status_message="Essential matrix RANSAC estimation failed",
                )

            # In case multiple Essential matrices were returned
            if E.shape != (3, 3):
                E = E[:3, :3]

            inliers = int(np.sum(mask))
            if inliers < self.min_features:
                return VisualTrackingResult(
                    success=False,
                    num_tracked=num_tracked,
                    num_inliers=inliers,
                    status_message=f"Insufficient geometric inliers ({inliers} < {self.min_features})",
                )

            # Recover relative pose: R and unit translation direction t
            _, R, t, mask_pose = cv2.recoverPose(
                E,
                pts_prev,
                pts_curr,
                cameraMatrix=self.K,
                mask=mask,
            )

            if R is None or t is None:
                return VisualTrackingResult(
                    success=False,
                    num_tracked=num_tracked,
                    num_inliers=inliers,
                    status_message="recoverPose failed to decompose Essential matrix",
                )

            # Verify R is a valid rotation matrix (det(R) ~ +1)
            det_R = np.linalg.det(R)
            if abs(det_R - 1.0) > 1e-2:
                return VisualTrackingResult(
                    success=False,
                    num_tracked=num_tracked,
                    num_inliers=inliers,
                    status_message=f"Invalid rotation matrix determinant: {det_R}",
                )

            t_unit = t.reshape(-1).astype(np.float64)
            t_norm = np.linalg.norm(t_unit)
            if t_norm > 1e-12:
                t_unit /= t_norm

            return VisualTrackingResult(
                success=True,
                num_tracked=num_tracked,
                num_inliers=inliers,
                R_rel=R.astype(np.float64),
                t_rel=t_unit,
                points_prev=pts_prev,
                points_curr=pts_curr,
                status_message="Relative motion successfully estimated",
            )

        except Exception as err:
            return VisualTrackingResult(
                success=False,
                num_tracked=num_tracked,
                num_inliers=0,
                status_message=f"OpenCV geometric estimation error: {err}",
            )
