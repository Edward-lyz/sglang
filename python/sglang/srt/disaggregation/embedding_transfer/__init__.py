"""
Embedding transfer stack for EPD disaggregation (prefill side).

Layered bottom-up; every import points downwards:

- ``base``               shared types, status enum and the transport interface
- ``embedding_data``     (package-external leaf) the pickled ZMQ payload type
- ``zmq_transport``      ZMQ transport implementation
- ``mooncake_transport`` Mooncake RDMA transport plus its pinned buffer pool
- ``request``            ``WaitingImageRequest``, the per-request state machine
- ``queues``             ``MMPreallocQueue`` / ``MMTransferQueue`` staged queues

This ``__init__`` deliberately re-exports nothing, so that importing any single
submodule does not drag in the whole stack (``request`` alone pulls
``managers.schedule_batch`` and both transports).  Import the submodule you need
directly.
"""
