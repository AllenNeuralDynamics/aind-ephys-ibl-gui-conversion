"""Make a lossless local SI/WavPack copy for read-only metric benchmarks."""

from pathlib import Path

import numpy as np
import spikeinterface as si
from benchmark_real_metric_pipeline import CASES, load_recording
from wavpack_numcodecs import WavPack


def main():
    """Write only to /tmp and verify representative raw samples on reload."""
    output = Path("/tmp/ibl-real-metric-benchmark/real_786867.zarr")
    original = load_recording(CASES["raw_786867"])
    if not output.exists():
        original.save_to_zarr(
            folder=output,
            compressor=WavPack(num_decoding_threads=1),
            channel_chunk_size=64,
            chunk_duration="1s",
            n_jobs=2,
            verbose=False,
            progress_bar=False,
        )
    reloaded = si.read_zarr(output)
    for start in (0, 9000000, 17970000):
        np.testing.assert_array_equal(
            original.get_traces(start_frame=start, end_frame=start + 30000),
            reloaded.get_traces(start_frame=start, end_frame=start + 30000),
        )
    for name in ("location", "group", "gain_to_uV", "offset_to_uV"):
        np.testing.assert_array_equal(
            original.get_property(name), reloaded.get_property(name)
        )
    print(f"Verified lossless samples and metadata: {output}")


if __name__ == "__main__":
    main()
