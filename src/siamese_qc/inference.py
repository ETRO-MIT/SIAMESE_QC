"""Three released QC methods for the femoral trochlear notch (V11 Knee landmark index 4)."""
import numpy as np
import torch
from .models import SiameseNet, SingleDistanceNet, ClassificationNet
from .assets import load_embeddings, Calibration


def load_model(weights, method, device='cpu'):
    """Load an original bare V11 state_dict strictly, without pretrained downloads."""
    factories = {'loe': SiameseNet,
                 'regression': SingleDistanceNet, 'classification': ClassificationNet}
    if method not in factories:
        raise ValueError(f'Unknown method: {method}')
    model = factories[method](128)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


class Predictor:
    """One checkpoint and its matching LoE library or classifier calibration."""
    def __init__(self, method, weights, *, metric='euclidean', reference=None,
                 calibration=None, device='cpu', k=None, threshold=None):
        if method not in ('loe','regression','classification'):
            raise ValueError('Supported methods: loe, regression, classification; RP is excluded because its private training references are not released')
        if metric not in ('euclidean','cosine'):
            raise ValueError('metric must be euclidean or cosine')
        if method in ('regression','classification') and metric != 'euclidean':
            raise ValueError('metric applies only to Siamese methods')
        self.method, self.metric, self.device = method, metric, torch.device(device)
        self.k = k if k is not None else 10
        if not isinstance(self.k, int) or isinstance(self.k, bool) or self.k < 1:
            raise ValueError('k must be a positive integer')
        defaults = {'loe': 1.29 if metric == 'euclidean' else 1.24,
                    'regression': 1.30, 'classification': 0.09}
        self.threshold = defaults[method] if threshold is None else float(threshold)
        if not np.isfinite(self.threshold):
            raise ValueError('threshold must be finite')
        self.reference = self.distances = None
        if method == 'loe':
            if reference is None:
                raise ValueError('LoE requires a matching Parquet embedding library')
            e, d = load_embeddings(reference)
            if len(e) < self.k:
                raise ValueError('LoE library contains fewer than K rows')
            self.reference = torch.from_numpy(e).to(self.device)
            self.distances = d
        elif reference is not None:
            raise ValueError('Reference libraries are only used by LoE')
        if calibration is not None and method != 'classification':
            raise ValueError('Only classification uses isotonic calibration')
        self.calibration = Calibration(calibration, 'logit_far') if calibration else None
        self.model = load_model(weights, method, self.device)

    @torch.inference_mode()
    def predict(self, patches, batch_size=32, *, allow_resampled_overshoot=False):
        """[N,3,128,128] in [0,1] -> numpy arrays; inaccurate=True means flag for review.

        Raw scores, calibrated distance and decision remain separate.
        LoE reports neighbor_distance_std_mm as population dispersion of neighbor
        distance labels (ddof=0), in millimeters.
        Classification decisions use sigmoid probabilities. The full-image pipeline
        enables allow_resampled_overshoot to preserve V11 interpolation values.
        """
        p = torch.as_tensor(patches, dtype=torch.float32)
        if p.ndim != 4 or tuple(p.shape[1:]) != (3,128,128) or len(p) == 0:
            raise ValueError('Expected a nonempty [N,3,128,128] tensor')
        if not torch.isfinite(p).all():
            raise ValueError('Patches must be finite')
        if not allow_resampled_overshoot and (p.min() < -1e-5 or p.max() > 1.00001):
            raise ValueError('Patches must be in [0,1]; allow_resampled_overshoot is only for normalized, B-spline-resampled inputs')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        scores, neighbors = [], []
        for batch in p.split(batch_size):
            outputs = self.model(batch.to(self.device))
            if self.method in ('regression','classification'):
                values = outputs[:,0].cpu().numpy()
                scores.extend(values * 18.0 if self.method == 'regression' else values)
                continue
            # Process targets independently to bound library comparison memory.
            for output in outputs:
                ref = self.reference
                if self.metric == 'euclidean':
                    similarities = torch.cdist(output[None], ref)[0]
                else:
                    similarities = torch.nn.functional.cosine_similarity(output[None], ref)
                ids = torch.argsort(similarities, descending=self.metric == 'cosine')[:self.k]
                distances = self.distances[ids.cpu().numpy()]
                neighbors.append(distances)
                scores.append(float(distances.mean()))
        score = np.asarray(scores, dtype=np.float32)
        result = {'score': score, 'inaccurate': score > self.threshold}
        if self.method in ('loe','regression'):
            result['distance_mm'] = score.copy()
        if self.method == 'classification':
            result['probability_inaccurate'] = score.copy()
        if neighbors:
            result['neighbor_distances_mm'] = np.stack(neighbors)
            result['neighbor_distance_std_mm'] = np.std(
                result['neighbor_distances_mm'], axis=1, ddof=0, dtype=np.float64
            )
        if self.calibration:
            values = score
            if self.method == 'classification':
                probability = np.clip(score.astype(float), 1e-6, 1-1e-6)
                values = np.log(probability / (1-probability))
            result['distance_mm'], result['distance_std_mm'] = self.calibration.predict_with_std(values)
        return result
