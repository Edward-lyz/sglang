"""
Wire data type for multimodal embeddings in EPD disaggregation mode.

``EmbeddingData`` is pickled by the encode side (``pull_zmq_server``,
``encode_server_aiak``) and unpickled by the prefill side, so it lives in a
leaf module that neither side's pipeline has to reach across to import.
"""

import torch


class EmbeddingData:
    """Data class for embedding results."""

    def __init__(
        self,
        req_id,
        num_parts,
        part_idx,
        image_grid_dim,
        embedding=None,
        error_msg=None,
        error_code=None,
    ):
        self.req_id = req_id
        self.num_parts = num_parts
        self.part_idx = part_idx
        self.image_grid_dim = image_grid_dim
        self.embedding = embedding
        self.send_time = None
        self.dtype = embedding.dtype if embedding is not None else None
        self.shape = list(embedding.shape) if embedding is not None else None
        # aggregated data
        self.ready_list = [i == self.part_idx for i in range(self.num_parts)]
        self.embedding_list = [
            embedding if i == self.part_idx else None for i in range(self.num_parts)
        ]
        self.image_grid_dim_list = [
            self.image_grid_dim if i == self.part_idx else None
            for i in range(self.num_parts)
        ]
        self.error_msg = error_msg
        self.error_code = error_code

    def add(self, embedding_data):
        """Add another embedding data part."""
        assert self.req_id == embedding_data.req_id
        assert not self.ready_list[embedding_data.part_idx]
        self.ready_list[embedding_data.part_idx] = True
        self.image_grid_dim_list[embedding_data.part_idx] = (
            embedding_data.image_grid_dim
        )
        self.embedding_list[embedding_data.part_idx] = embedding_data.embedding

    def get_embedding(self, is_concat=False):
        """Get embedding data, optionally concatenated."""
        if is_concat:
            return torch.concat([embedding.cuda() for embedding in self.embedding_list])
        else:
            return self.embedding_list

    def get_img_grid(self):
        """Get concatenated image grid dimensions."""
        return torch.concatenate(self.image_grid_dim_list)

    @property
    def ready(self):
        """Check if all parts are ready."""
        return sum(self.ready_list) == self.num_parts

    def __repr__(self):
        """Return string representation."""
        return (
            f"EmbeddingData(req_id={self.req_id}, num_parts={self.num_parts}, "
            f"part_idx={self.part_idx}) error_msg={self.error_msg}"
        )

    def copy_without_embedding(self):
        """Create a copy without embedding data."""
        new_data = EmbeddingData(
            req_id=self.req_id,
            num_parts=self.num_parts,
            part_idx=self.part_idx,
            image_grid_dim=self.image_grid_dim,
            error_msg=self.error_msg,
            error_code=self.error_code,
        )
        new_data.send_time = self.send_time
        new_data.dtype = self.dtype
        new_data.shape = self.shape
        return new_data
