"""Green-context partitions used by DSV4 shared-expert overlap."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GreenContextPartition:
    """Complementary green-context partitions for one communication window.

    ``compute_stream`` runs compute kernels on ``compute_sms`` SMs;
    ``comm_stream`` runs the main-stream kernels beside them on the others.
    """

    compute_stream: torch.cuda.Stream
    comm_stream: torch.cuda.Stream
    compute_sms: int


SM_GROUP_SIZE = 8
# Device -> partition pairs and owning contexts, retained for process lifetime.
_DEVICE_PARTITIONS: dict[int, tuple[list[GreenContextPartition | None], list]] = {}


def get_or_create_partition_pairs(
    device: torch.device,
) -> list[GreenContextPartition | None]:
    """One window per shared-partition size, built before graph capture.

    The device is split once into SM groups; the window with ``k`` groups
    gives its shared partition the first ``k`` groups and its comm
    partition the rest, so windows of different sizes nest.
    """
    from cuda.bindings import driver

    def call(result: tuple) -> tuple:
        if result[0] != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"green context setup failed: {result[0]}")
        return result[1:]

    index = device.index if device.index is not None else torch.cuda.current_device()
    cached = _DEVICE_PARTITIONS.get(index)
    if cached is not None:
        return cached[0]

    contexts = []
    stream_handles = []

    def green_stream(resources: list) -> torch.cuda.Stream:
        (desc,) = call(driver.cuDevResourceGenerateDesc(resources, len(resources)))
        (green_ctx,) = call(
            driver.cuGreenCtxCreate(
                desc,
                index,
                driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
            )
        )
        contexts.append(green_ctx)
        (handle,) = call(
            driver.cuGreenCtxStreamCreate(
                green_ctx, driver.CUstream_flags.CU_STREAM_NON_BLOCKING, 0
            )
        )
        stream_handles.append(handle)
        return torch.cuda.ExternalStream(int(handle), device=device)

    (sm_resource,) = call(
        driver.cuDeviceGetDevResource(
            index, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )
    groups, num_groups, leftover = call(
        driver.cuDevSmResourceSplitByCount(
            sm_resource.sm.smCount // SM_GROUP_SIZE,
            sm_resource,
            0,
            SM_GROUP_SIZE,
        )
    )
    groups = list(groups[:num_groups])
    windows: list[GreenContextPartition | None] = [None]
    try:
        for shared_groups in range(1, num_groups + 1):
            comm_resources = groups[shared_groups:]
            if leftover.sm.smCount > 0:
                comm_resources.append(leftover)
            if not comm_resources:
                break
            compute_stream = green_stream(groups[:shared_groups])
            comm_stream = green_stream(comm_resources)
            windows.append(
                GreenContextPartition(
                    compute_stream, comm_stream, shared_groups * SM_GROUP_SIZE
                )
            )
    except Exception:
        # ExternalStream does not own its handle. Release partial setup
        # before a later layer retries, preserving the original failure.
        for handle in reversed(stream_handles):
            driver.cuStreamDestroy(handle)
        for context in reversed(contexts):
            driver.cuGreenCtxDestroy(context)
        raise
    _DEVICE_PARTITIONS[index] = (windows, contexts)
    return windows
