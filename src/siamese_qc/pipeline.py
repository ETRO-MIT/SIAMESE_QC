"""Full-radiograph input -> prepared knee patches -> QC predictions."""
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from .preprocessing import prepare_image, extract_patches


def estimate_spacing_from_condyles(condyles_xy, *, reference_width_mm=53.90):
    """Paper's bilateral anatomical calibration -> isotropic (sx,sy) mm/pixel.

    condyles_xy has shape [2,2,2]: two endpoints of the femoral condylar line
    for each knee, in the original radiograph pixel frame. Knee/endpoint order
    does not affect the result. Assumes isotropic source pixels. The default
    reference width is the paper's training-set median intercondylar width.
    """
    points = np.asarray(condyles_xy, dtype=float)
    if points.shape != (2,2,2) or not np.isfinite(points).all():
        raise ValueError('condyles_xy must contain two finite (x,y) endpoints per knee: [2,2,2]')
    if not np.isfinite(reference_width_mm) or reference_width_mm <= 0:
        raise ValueError('reference_width_mm must be finite and positive')
    widths = np.linalg.norm(points[:,1]-points[:,0],axis=1)
    if (widths <= 0).any():
        raise ValueError('Condylar endpoints must be distinct for both knees')
    spacing = float(reference_width_mm/widths.mean())
    return (spacing,spacing)


def read_radiograph(source, *, spacing_mm=None):
    """Read one scalar radiograph, optionally overriding (x,y) spacing in mm.

    Accept a path or SimpleITK image. Singleton-z and legacy three-identical-slice images are reduced to 2-D;
    true volumes, vector/RGB images and DICOM series directories are rejected.
    Raster formats without calibrated spacing require spacing_mm explicitly.
    """
    if isinstance(source, sitk.Image):
        image = sitk.Image(source)
    else:
        path = Path(source)
        if path.is_dir():
            raise ValueError('Supply one radiograph file, not a DICOM series directory')
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        image_io = reader.GetImageIOFromFileName(str(path))
        if not image_io:
            raise ValueError('Unsupported or unreadable radiograph format')
        reader.ReadImageInformation()
        raster_io = {'PNGImageIO','JPEGImageIO','TIFFImageIO','BMPImageIO'}
        if spacing_mm is None and image_io in raster_io:
            raise ValueError('Raster radiographs require explicit spacing_mm or condyle calibration in millimeters')
        if spacing_mm is None and image_io == 'GDCMImageIO' and not reader.HasMetaDataKey('0028|0030'):
            raise ValueError('DICOM PixelSpacing is missing; supply spacing_mm or condyle calibration')
        image = reader.Execute()
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('Supply a scalar grayscale radiograph, not RGB/vector data')
    if image.GetDimension() == 3:
        depth = image.GetSize()[2]
        if depth == 3:
            channels = sitk.GetArrayFromImage(image)
            if not (np.array_equal(channels[0],channels[1]) and
                    np.array_equal(channels[0],channels[2])):
                raise ValueError('Three-slice radiographs must contain identical grayscale channels, not a volume')
            image = image[:, :, 0]
        elif depth == 1:
            image = image[:, :, 0]
    if image.GetDimension() != 2:
        raise ValueError('Supply a 2-D radiograph, singleton-z image, or three identical grayscale slices')
    if spacing_mm is not None:
        spacing = np.asarray(spacing_mm, dtype=float)
        if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError('spacing_mm must be two finite positive values (x,y)')
        image.SetSpacing(tuple(spacing))
    if not np.isfinite(image.GetSpacing()).all() or min(image.GetSpacing()) <= 0:
        raise ValueError('Image spacing must be finite positive millimeters per pixel')
    return image


def prepare_radiograph(source, points, knee_boxes, sides, *, intensity_divisor,
                       spacing_mm=None):
    """Return [N,3,128,128] patches and preprocessing geometry.

    points: [N,2] (x,y) pixel coordinates in the ORIGINAL supplied image.
    knee_boxes: [N,4] (xmin,ymin,xmax,ymax), upper bounds exclusive;
                one [4] box may be shared by all candidates in the same knee.
    sides: N values 'right'/'left', or one string for a shared knee.

    Each row assesses the femoral trochlear notch, not arbitrary landmarks.
    The full image is divided by an explicit intensity divisor and resampled
    to 0.2 mm/pixel before cropping (as in the research preprocessing). Each
    knee is cropped and left knees mirrored before patch extraction. Crops
    are transient; only patches/geometry are returned. B-spline overshoots
    are preserved as in V11. No image-dependent min/max scaling is performed.
    """
    image = read_radiograph(source, spacing_mm=spacing_mm)
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all():
        raise ValueError('points must be nonempty finite [N,2] source pixel coordinates')
    boxes = np.asarray(knee_boxes, dtype=float)
    if boxes.shape == (4,):
        boxes = np.tile(boxes, (len(p),1))
    if boxes.shape != (len(p),4) or not np.isfinite(boxes).all():
        raise ValueError('knee_boxes must be [4] or [N,4] finite source pixel bounds')
    sides = [sides]*len(p) if isinstance(sides,str) else list(sides)
    if len(sides) != len(p) or any(side not in ('right','left') for side in sides):
        raise ValueError('Supply right/left for each candidate, or one shared side')
    size = np.asarray(image.GetSize())
    if ((boxes[:,:2] < 0).any() or (boxes[:,2:] > size).any() or
            (boxes[:,2:] <= boxes[:,:2]).any()):
        raise ValueError('Knee boxes must have positive extent inside the original radiograph')
    if ((p < boxes[:,:2]).any() or (p >= boxes[:,2:]).any()):
        raise ValueError('Each landmark center must lie inside its supplied knee bounding box')
    # The same image-wide resampling grid is used for all knees/candidates.
    array, mapped = prepare_image(image, p, intensity_divisor=intensity_divisor,
                                  side='right', allow_resampled_overshoot=True)
    # V11 uses TransformPhysicalPointToIndex (nearest pixel), not truncation
    # of a continuous coordinate when selecting the patch center.
    continuous_mapped = mapped.copy()
    mapped = np.floor(mapped + 0.5).astype(int)
    scale = np.asarray(image.GetSpacing())/0.2
    bounds = np.floor(boxes * np.tile(scale,2)+0.5).astype(int)
    bounds[:,:2] = np.maximum(bounds[:,:2],0)
    bounds[:,2:] = np.minimum(bounds[:,2:], np.asarray(array.shape[::-1]))
    patches, centers, geometries = [], [], []
    for point, box, side in zip(mapped, bounds, sides):
        x1,y1,x2,y2 = box.tolist()
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Knee box has no pixels after resampling')
        crop = array[y1:y2,x1:x2]
        center = point-np.asarray([x1,y1])
        if side == 'left':
            crop = crop[:,::-1].copy()
            center[0] = crop.shape[1]-1-center[0]
        if (center < 0).any() or (center > np.asarray(crop.shape[::-1])-1).any():
            raise ValueError('Landmark falls outside the resampled crop pixel centers; review a boundary-touching box/point')
        patches.append(extract_patches(crop, [center], allow_resampled_overshoot=True)[0])
        centers.append(center)
        geometries.append([x1,y1,x2,y2])
    import torch
    return {
        'patches': torch.stack(patches),
        'source_points_xy': p.copy(),
        'source_boxes_xyxy': boxes.copy(),
        'source_spacing_mm': np.asarray(image.GetSpacing()),
        'resampled_points_xy': mapped,
        'resampled_continuous_points_xy': continuous_mapped,
        'resampled_boxes_xyxy': np.asarray(geometries),
        'prepared_crop_points_xy': np.asarray(centers),
        'sides': sides,
    }


def predict_radiograph(predictor, source, points, knee_boxes, sides, *,
                        intensity_divisor, spacing_mm=None, batch_size=32):
    """End-to-end inference; preserve input order and return QC plus geometry."""
    prepared = prepare_radiograph(source, points, knee_boxes, sides,
                                  intensity_divisor=intensity_divisor, spacing_mm=spacing_mm)
    results = predictor.predict(prepared['patches'], batch_size=batch_size, allow_resampled_overshoot=True)
    return {'predictions': results, 'preprocessing': prepared}
