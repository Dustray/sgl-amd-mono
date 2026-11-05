from collections import Counter
import hashlib


class WeightObservation:
    def __init__(self):
        self.count = 0
        self.total_bytes = 0
        self.completed = False
        self.dtypes = Counter()
        self.names_hash = hashlib.sha256()

    def iterate(self, weights):
        for name, tensor in weights:
            self.count += 1
            self.total_bytes += tensor.numel() * tensor.element_size()
            self.dtypes[str(tensor.dtype)] += 1
            self.names_hash.update(name.encode() + b"\0")
            yield name, tensor
        self.completed = True

    def summary(self):
        return {
            "tensor_count": self.count,
            "total_bytes": self.total_bytes,
            "dtypes": dict(self.dtypes),
            "tensor_names_sha256": self.names_hash.hexdigest(),
            "completed": self.completed,
            "retained_tensor_copies": 0,
        }
