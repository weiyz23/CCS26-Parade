from .euclidean import EuclideanRoutingModel
from .hyperbolic import HyperbolicRoutingModel


def build_model(geometry: str, num_as: int, num_profiles: int, embed_dim: int, num_vectors: int):
    if geometry == "hyperbolic":
        return HyperbolicRoutingModel(
            num_as=num_as,
            num_profiles=num_profiles,
            embed_dim=embed_dim,
            num_vectors=num_vectors,
        )
    if geometry == "euclidean":
        return EuclideanRoutingModel(
            num_as=num_as,
            num_profiles=num_profiles,
            embed_dim=embed_dim,
            num_vectors=num_vectors,
        )
    raise ValueError(f"Unsupported geometry: {geometry}")
