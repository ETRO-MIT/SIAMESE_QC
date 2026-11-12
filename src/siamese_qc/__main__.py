"""CLI for full-radiograph QC or an already prepared knee crop."""
import argparse
import csv
from pathlib import Path
from .inference import Predictor
from .preprocessing import read_prepared_crop, extract_patches
from .pipeline import prepare_radiograph, estimate_spacing_from_condyles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=['loe','regression','classification'])
    parser.add_argument('--weights', required=True)
    parser.add_argument('--image', required=True, help='Full scalar radiograph (default), or prepared crop with --input-mode prepared')
    parser.add_argument('--points', required=True, help='CSV: x,y,xmin,ymin,xmax,ymax,side in original radiograph pixels; prepared mode requires only x,y')
    parser.add_argument('--input-mode', choices=['radiograph','prepared'], default='radiograph')
    parser.add_argument('--intensity-divisor', type=float, help='Required for radiograph mode: 255 for research 8-bit images; 1 for [0,1] data')
    spacing_options = parser.add_mutually_exclusive_group()
    spacing_options.add_argument('--spacing', type=float, nargs=2, metavar=('SX','SY'), help='Source spacing in mm/pixel; raster images need this or --condyles')
    spacing_options.add_argument('--condyles', type=float, nargs=8, metavar='PIXEL',
                                 help='Missing spacing: RX1 RY1 RX2 RY2 LX1 LY1 LX2 LY2, source-image pixel coordinates of condylar endpoints')
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference', help='LoE embedding library (.parquet)')
    parser.add_argument('--calibration', help='Optional single/ensemble classifier calibration JSON (distance mean/std)')
    parser.add_argument('--metric', choices=['euclidean','cosine'], default='euclidean')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--k', type=int)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    source_spacing = args.spacing
    if args.condyles is not None:
        import numpy as np
        source_spacing = estimate_spacing_from_condyles(np.asarray(args.condyles).reshape(2,2,2))
    with open(args.points, newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {'x','y'}.issubset(rows[0]):
        parser.error('points CSV must have x,y columns and at least one row')
    points = [[float(row['x']), float(row['y'])] for row in rows]
    prepared = None
    if args.input_mode == 'radiograph':
        needed = {'xmin','ymin','xmax','ymax','side'}
        if not needed.issubset(rows[0]):
            parser.error('Radiograph mode requires xmin,ymin,xmax,ymax,side columns in the points CSV')
        if args.intensity_divisor is None:
            parser.error('Radiograph mode requires --intensity-divisor')
        boxes = [[float(row[key]) for key in ('xmin','ymin','xmax','ymax')] for row in rows]
        prepared = prepare_radiograph(args.image, points, boxes, [row['side'] for row in rows],
                                     intensity_divisor=args.intensity_divisor, spacing_mm=source_spacing)
        patches = prepared['patches']
    else:
        if source_spacing is not None or args.intensity_divisor is not None:
            parser.error('Prepared mode already has 0.2 mm/pixel [0,1] data; omit --spacing, --condyles and --intensity-divisor')
        patches = extract_patches(read_prepared_crop(args.image), points, allow_resampled_overshoot=True)
    predictor = Predictor(args.method, args.weights, metric=args.metric, reference=args.reference,
                          calibration=args.calibration, device=args.device, k=args.k,
                          threshold=args.threshold)
    results = predictor.predict(patches, batch_size=args.batch_size, allow_resampled_overshoot=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve input columns and row order; include method/metric for interpretation.
    output_rows = []
    for index, row in enumerate(rows):
        record = dict(row, method=args.method, metric=args.metric, threshold=predictor.threshold)
        if prepared is not None:
            center = prepared['prepared_crop_points_xy'][index]
            record.update(qc_crop_x=float(center[0]), qc_crop_y=float(center[1]),
                          source_spacing_x_mm=float(prepared['source_spacing_mm'][0]),
                          source_spacing_y_mm=float(prepared['source_spacing_mm'][1]),
                          spacing_source='condyle_estimate' if args.condyles is not None else ('explicit' if args.spacing is not None else 'metadata'))
        for name, values in results.items():
            if name != 'neighbor_distances_mm':
                record[name] = values[index].item()
        output_rows.append(record)
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == '__main__':
    main()
