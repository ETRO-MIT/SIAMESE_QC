"""Portable reference assets and inference-only isotonic interpolation."""
import json
from pathlib import Path
import numpy as np


EMBEDDING_COLUMNS = [f'embedding_{i:03d}' for i in range(128)]
DISTANCE_COLUMN = 'distance_mm'


def validate_embeddings(embeddings, distances):
    """Validate normalized 128-dimensional vectors and aligned distance labels."""
    if embeddings.ndim != 2 or embeddings.shape[1] != 128 or not len(embeddings):
        raise ValueError('Expected nonempty embeddings [M,128]')
    if not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-3):
        raise ValueError('Library embeddings must be finite and L2 normalized')
    if distances.shape != (len(embeddings),) or not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError('Expected finite nonnegative distance_mm labels [M]')


def load_embeddings(path):
    """Load canonical or original V11 Parquet embedding/distance columns.

    Named column selection preserves vector order regardless of Parquet column
    order. No subject identifiers or patch coordinates are needed.
    """
    import pyarrow.parquet as pq
    if Path(path).suffix.lower() != '.parquet':
        raise ValueError('LoE library must be a .parquet file')
    schema = pq.read_schema(path)
    if set([DISTANCE_COLUMN] + EMBEDDING_COLUMNS).issubset(schema.names):
        distance_column, embedding_columns = DISTANCE_COLUMN, EMBEDDING_COLUMNS
    else:
        # Existing V11 Parquet exports retain research column names. Select by
        # explicit numeric names, never by position or lexicographic sorting.
        distance_column = 'Distance [mm]'
        embedding_columns = [f'Emb col_{i}' for i in range(1,129)]
        if not set([distance_column] + embedding_columns).issubset(schema.names):
            raise ValueError('Library must use canonical distance_mm/embedding_000...127 or V11 Distance [mm]/Emb col_1...128 columns')
    columns = [distance_column] + embedding_columns
    table = pq.read_table(path, columns=columns)
    if any(table[name].null_count for name in columns):
        raise ValueError('LoE library must not contain nulls')
    embeddings = np.column_stack([table[name].to_numpy() for name in embedding_columns]).astype(np.float32)
    distances = table[distance_column].to_numpy().astype(np.float32)
    validate_embeddings(embeddings, distances)
    return embeddings, distances


class Calibration:
    """Portable single isotonic mapping or ensemble of fitted mappings.

    Member predictions use clipped linear interpolation. Ensembles reproduce
    the HTO notebook's mean and population standard deviation (ddof=0).
    """
    def __init__(self, path, expected_score_type):
        data = json.loads(Path(path).read_text())
        if data['score_type'] != expected_score_type:
            raise ValueError('Calibration score_type does not match the predictor')
        if data.get('logit_clip_epsilon', 1e-6) != 1e-6:
            raise ValueError('Classifier calibration must use logit clipping epsilon 1e-6')
        if 'models' in data:
            if data.get('aggregation') != 'mean' or data.get('std_ddof') != 0:
                raise ValueError('Ensembles require mean aggregation and population std (ddof=0)')
            models = data['models']
            if not isinstance(models, list) or not models:
                raise ValueError('Calibration ensemble must contain at least one model')
        else:
            models = [data]
        self.members = []
        for member in models:
            if member.get('score_type', expected_score_type) != expected_score_type:
                raise ValueError('Ensemble members must use the same score_type')
            x = np.asarray(member['x_thresholds'], dtype=float)
            y = np.asarray(member['y_thresholds_mm'], dtype=float)
            if (x.ndim != 1 or x.size == 0 or y.shape != x.shape or
                not np.isfinite(x).all() or not np.isfinite(y).all() or
                (np.diff(x) <= 0).any() or (y < 0).any()):
                raise ValueError('Invalid isotonic thresholds')
            differences = np.diff(y)
            if (differences > 0).any() and (differences < 0).any():
                raise ValueError('Isotonic outputs must be monotonic')
            if expected_score_type == 'logit_far' and (differences < 0).any():
                raise ValueError('Classifier distance must increase with logit_far')
            self.members.append((x,y))

    def predict_members(self, scores):
        scores = np.asarray(scores, dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError('Calibration scores must be finite')
        return np.stack([np.interp(scores,x,y) for x,y in self.members])

    def predict_with_std(self, scores):
        values = self.predict_members(scores)
        return np.mean(values,axis=0), np.std(values,axis=0,ddof=0)

    def predict(self, scores):
        return self.predict_with_std(scores)[0]
