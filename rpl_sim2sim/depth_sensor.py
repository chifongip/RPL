"""Depth consumer for the independently launched local sim2sim processes."""

from multiprocessing import resource_tracker

from holosoma_inference.sensors.depth_shm import DepthShmSensor


class LocalDepthConsumer(DepthShmSensor):
    def _adopt(self, shm):
        super()._adopt(shm)
        # Python <=3.12 tracks even read-only attachments as owners. The
        # independently launched producer owns unlinking; consumer exit must
        # not unlink its block while simulation continues.
        resource_tracker.unregister(shm._name, "shared_memory")
