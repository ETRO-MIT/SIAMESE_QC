"""Image and coordinate handling for 0.2 mm/pixel knee crops."""
import numpy as np
import torch
import SimpleITK as sitk


def extract_patches(image, points, *, allow_resampled_overshoot=False):
    """Prepared [H,W], [1,H,W], or [3,H,W] image, (x,y) crop pixels -> [N,3,128,128].

    No resizing, intensity scaling, or flipping is performed here. Borders use
    image-minimum padding, matching V11 inspect_and_pad. Starts use int(x-64).
    """
    a = np.asarray(image, dtype=np.float32)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3 or a.shape[0] not in (1, 3) or min(a.shape[1:]) == 0:
        raise ValueError('Image must have shape [H,W], [1,H,W], or [3,H,W]')
    if not np.isfinite(a).all():
        raise ValueError('Prepared image must be finite')
    if not allow_resampled_overshoot and (a.min() < -1e-5 or a.max() > 1.00001):
        raise ValueError('Prepared image must be in [0,1]; enable allow_resampled_overshoot only for normalized, B-spline-resampled images')
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all():
        raise ValueError('points must be a nonempty finite [N,2] array of (x,y) pixels')
    h, w = a.shape[1:]
    if ((p < 0).any() or (p[:, 0] > w-1).any() or (p[:, 1] > h-1).any()):
        raise ValueError('Landmark centers must lie inside the image')
    if a.shape[0] == 1:
        a = np.repeat(a, 3, axis=0)
    patches = []
    for x, y in p:
        # V11 adds an integer margin before int() for border patches.
        border = not (128 <= x < w-128 and 128 <= y < h-128)
        pad = 128 if border else 0
        padded = np.pad(a, ((0,0),(pad,pad),(pad,pad)), constant_values=float(a.min())) if pad else a
        x1, y1 = int(x + pad - 64), int(y + pad - 64)
        patches.append(padded[:, y1:y1+128, x1:x1+128])
    return torch.from_numpy(np.stack(patches).copy())


def read_prepared_crop(path):
    """Read 2-D scalar or legacy scalar [3,H,W] MHA; verify in-plane spacing."""
    image = sitk.ReadImage(str(path))
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('Expected a scalar image, not a vector/RGB image')
    if not np.allclose(image.GetSpacing()[:2], (0.2, 0.2), atol=1e-6):
        raise ValueError('Prepared crops must have 0.2 mm/pixel in-plane spacing')
    return sitk.GetArrayFromImage(image)


def prepare_image(image, points, *, intensity_divisor, side='right', clip_resampled=False, allow_resampled_overshoot=False):
    """Resample a 2-D scalar ROI and map its source pixel points to QC pixels.

    Use divisor=255 for the research 8-bit images, or 1 for already [0,1]
    intensities. No DICOM windowing/inversion or ROI detection is inferred.
    Left knees are flipped once after resampling. Returns array, mapped points.
    Set allow_resampled_overshoot=True to preserve B-spline overshoots (V11),
    or clip_resampled=True to clip them. Default rejects them for low-level use.
    """
    if image.GetDimension() != 2 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('Supply a 2-D scalar knee ROI')
    if side not in ('right', 'left') or not np.isfinite(intensity_divisor) or intensity_divisor <= 0:
        raise ValueError('Use side right/left and a finite positive intensity_divisor')
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all():
        raise ValueError('points must have finite shape [N,2]')
    size = np.asarray(image.GetSize())
    if (p < 0).any() or (p > size-1).any():
        raise ValueError('Source points must lie inside the ROI')
    a = sitk.GetArrayFromImage(image).astype(np.float32) / intensity_divisor
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError('Divided intensities must lie in [0,1]')
    normalized = sitk.GetImageFromArray(a)
    normalized.CopyInformation(image)
    out_size = np.round(size * np.asarray(image.GetSpacing()) / 0.2).astype(int)
    if (out_size < 1).any():
        raise ValueError('ROI is too small to resample')
    resampled = sitk.Resample(normalized, out_size.tolist(), sitk.Transform(), sitk.sitkBSpline,
                              image.GetOrigin(), (0.2,0.2), image.GetDirection(), 0., sitk.sitkFloat32)
    mapped = np.asarray([resampled.TransformPhysicalPointToContinuousIndex(
        image.TransformContinuousIndexToPhysicalPoint(tuple(v))) for v in p])
    a = sitk.GetArrayFromImage(resampled)
    if clip_resampled:
        a = np.clip(a, 0., 1.)
    # The low-level API rejects overshoots unless clipping is explicitly enabled.
    if not allow_resampled_overshoot and (a.min() < -1e-5 or a.max() > 1.00001):
        raise ValueError('B-spline intensities overshot [0,1]; enable allow_resampled_overshoot to preserve the V11 convention')
    if side == 'left':
        a = a[:, ::-1].copy()
        mapped[:, 0] = a.shape[1]-1-mapped[:, 0]
    return a, mapped
