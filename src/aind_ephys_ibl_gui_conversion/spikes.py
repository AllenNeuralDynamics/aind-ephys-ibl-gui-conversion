"""Extract spike data from sorting results."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
from packaging.version import Version
from spikeinterface.exporters import export_to_phy

from aind_ephys_ibl_gui_conversion.recording_utils import (
    _stream_to_probe_name,
)

# Metric names that were renamed in spikeinterface >= 0.104
_DEPRECATED_METRIC_RENAMES = {
    "peak_to_valley": "peak_to_trough_duration",
    "peak_trough_ratio": "waveform_ratios",
    "num_positive_peaks": "waveform_baseline_flatness",
    "num_negative_peaks": "waveform_baseline_flatness",
    "velocity_above": "velocity_fits",
    "velocity_below": "velocity_fits",
}


def _patch_si_deprecated_metric_validation() -> None:
    """Monkey-patch spikeinterface >= 0.104 to ignore deprecated template
    metric names when loading legacy waveform extractor folders.

    In Code Ocean the data folders are immutable so we cannot rewrite
    params.json in place. Instead we patch BaseMetricExtension._set_params,
    which is where the ValueError is raised for deprecated metric names.
    The remapping here is intentionally minimal — the full cleanup of metric
    params keys, velocity sub-params, etc. is handled afterward by
    ComputeTemplateMetrics._handle_backward_compatibility_on_load.
    """
    si_version = Version(si.__version__)
    if si_version < Version("0.104.0"):
        return

    try:
        from spikeinterface.core import analyzer_extension_core as _aec

        original_set_params = _aec.BaseMetricExtension._set_params

        def _patched_set_params(self, metric_names=None, **kwargs):
            if metric_names is not None:
                metric_names = [
                    _DEPRECATED_METRIC_RENAMES.get(n, n)
                    for n in metric_names
                ]
                # Deduplicate (e.g. velocity_above + velocity_below -> velocity_fits)
                seen = []
                for n in metric_names:
                    if n not in seen:
                        seen.append(n)
                metric_names = seen

            return original_set_params(self, metric_names=metric_names, **kwargs)

        _aec.BaseMetricExtension._set_params = _patched_set_params
        logging.info(
            "Patched spikeinterface deprecated template metric validation."
        )
    except Exception as e:
        logging.warning(
            f"Could not patch spikeinterface metric validation: {e}"
        )



def extract_spikes(  # noqa: C901
    sorting_folder: Path,
    results_folder: Path,
    stream_to_use: str | None = None,
    min_duration_secs: int = 300,
):
    """Extract spike data from a sorting folder.

    Parameters
    ----------
    sorting_folder : Path
        Path to the sorted spike data folder.
    results_folder : Path
        Path where extracted spike data will be saved.
    stream_to_use : str or None
        If provided, only process this stream.
    min_duration_secs : int
        Minimum duration (seconds) for spike extraction.
    """
    # Must be called here as well as module level — this function runs in a
    # subprocess via concurrent.futures and the module-level patch won't
    # carry over to the child process.
    _patch_si_deprecated_metric_validation()

    session_folder = Path(str(sorting_folder).split("_sorted")[0])
    scratch_folder = Path("/scratch")

    ecephys_folder = session_folder / "ecephys_clipped"
    if ecephys_folder.is_dir():
        ecephys_compressed_folder = session_folder / "ecephys_compressed"
    else:
        ecephys_folder = session_folder / "ecephys" / "ecephys_clipped"
        ecephys_compressed_folder = (
            session_folder / "ecephys" / "ecephys_compressed"
        )
    print(f"ecephys folder: {ecephys_folder}")
    print(f"ecephys compressed folder: {ecephys_compressed_folder}")

    postprocessed_folder = sorting_folder / "postprocessed"

    stream_names, stream_ids = se.get_neo_streams(
        "openephysbinary", ecephys_folder
    )

    neuropix_streams = [s for s in stream_names if "Neuropix" in s]
    probe_names = [_stream_to_probe_name(s) for s in neuropix_streams]

    if stream_to_use is not None:
        logging.info(
            "Stream name provided as parameter. Will only process "
            f"{stream_to_use}"
        )

    for idx, stream_name in enumerate(neuropix_streams):
        if stream_to_use is not None and stream_name != stream_to_use:
            continue

        analyzer_mappings = []
        num_shanks = 0
        shank_glob = tuple(postprocessed_folder.glob(f"*{stream_name}*group*"))
        if shank_glob:
            num_shanks = len(shank_glob)

        print("Number of shanks", num_shanks)

        if "-LFP" in stream_name:
            continue

        print(stream_name)

        probe_name = probe_names[idx]

        output_folder = Path(results_folder) / probe_name

        if not output_folder.is_dir():
            output_folder.mkdir()

        print("Loading sorting analyzer...")
        if num_shanks > 1:
            for shank_index in range(num_shanks):
                analyzer_folder = (
                    postprocessed_folder / f"experiment1_{stream_name}_"
                    f"recording1_group{shank_index}.zarr"
                )

                if analyzer_folder.is_dir():
                    analyzer = si.load_sorting_analyzer(analyzer_folder)
                else:
                    analyzer_folder = (
                        postprocessed_folder / f"experiment1_{stream_name}_"
                        f"recording1_group{shank_index}"
                    )
                    if not analyzer_folder.exists():
                        with open(
                            output_folder / "sorting_error.txt", "w"
                        ) as f:
                            f.write(
                                "No postprocessed sorting "
                                f"output found for {probe_name}"
                            )
                        continue

                    analyzer = si.load_sorting_analyzer_or_waveforms(
                        analyzer_folder
                    )

                if analyzer.get_total_duration() < min_duration_secs:
                    continue

                analyzer_mappings.append(analyzer)
        else:
            analyzer_folder = (
                postprocessed_folder
                / f"experiment1_{stream_name}_recording1.zarr"
            )
            if analyzer_folder.is_dir():
                analyzer = si.load_sorting_analyzer(analyzer_folder)
            else:
                analyzer_folder = (
                    postprocessed_folder
                    / f"experiment1_{stream_name}_recording1"
                )
                if not analyzer_folder.exists():
                    with open(output_folder / "sorting_error.txt", "w") as f:
                        f.write(
                            "No postprocessed sorting output "
                            f"found for {probe_name}"
                        )
                    continue

                analyzer = si.load_sorting_analyzer_or_waveforms(
                    analyzer_folder
                )
            analyzer_mappings.append(analyzer)

        phy_folder = scratch_folder / f"{postprocessed_folder.parent.name}_phy"

        print("Exporting to phy format...")

        spike_depths = []
        clusters = []
        spike_samples = []
        amps = []
        shank_indices = []
        cluster_channels = []
        unit_shank_indices = []
        cluster_peak_to_trough = []
        cluster_waveforms = []

        templates = []
        quality_metrics = []

        for index, analyzer in enumerate(analyzer_mappings):
            export_to_phy(
                analyzer,
                output_folder=phy_folder,
                compute_pc_features=False,
                remove_if_exists=True,
                copy_binary=False,
                dtype="int16",
            )

            spike_locations = analyzer.get_extension(
                "spike_locations"
            ).get_data()
            template_ext = analyzer.get_extension("templates")
            templates = template_ext.get_templates()

            for unit_idx, unit_id in enumerate(analyzer.unit_ids):
                waveform = templates[unit_idx, :, :]
                peak_channel = np.argmax(
                    np.max(waveform, 0) - np.min(waveform, 0)
                )
                peak_waveform = waveform[:, peak_channel]
                peak_to_trough = (
                    np.argmax(peak_waveform) - np.argmin(peak_waveform)
                ) / 30000.0
                cluster_channels.append(peak_channel)
                unit_shank_indices.append(index)
                cluster_peak_to_trough.append(peak_to_trough)
                cluster_waveforms.append(waveform)

            print("Converting data...")

            current_clusters = np.load(phy_folder / "spike_clusters.npy")
            clusters.append(current_clusters)

            for cluster in current_clusters:
                shank_indices.append(index)

            spike_samples.append(np.load(phy_folder / "spike_times.npy"))
            amps.append(np.load(phy_folder / "amplitudes.npy"))
            spike_depths.append(spike_locations["y"])

            qm = analyzer.get_extension("quality_metrics")
            qm_data = qm.get_data()

            qm_data.index.name = "cluster_id"
            qm_data["cluster_id.1"] = qm_data.index.values
            if "default_qc" in analyzer.sorting.get_property_keys():
                qm_data["default_qc"] = analyzer.sorting.get_property(
                    "default_qc"
                )

            if (
                "decoder_label" in analyzer.sorting.get_property_keys()
                or "unitrefine_label" in analyzer.sorting.get_property_keys()
            ):
                unitrefine_column_name = (
                    "decoder_label"
                    if "decoder_label" in analyzer.sorting.get_property_keys()
                    else "unitrefine_label"
                )
                qm_data["unitrefine_label"] = analyzer.sorting.get_property(
                    unitrefine_column_name
                )

            quality_metrics.append(qm_data)

        if len(analyzer_mappings) == 1:
            spike_clusters = np.squeeze(clusters[0].astype("uint32"))
            spike_times = np.squeeze(spike_samples[0] / 30000.0).astype(
                "float64"
            )
            spike_amps = np.squeeze(-amps[0]).astype("float64")
            spike_depths_array = spike_depths[0]
            quality_metrics_df = quality_metrics[0]
        else:
            spike_clusters = np.squeeze(
                np.concatenate(clusters).astype("uint32")
            )
            spike_times = np.squeeze(
                np.concatenate(spike_samples) / 30000.0
            ).astype("float64")
            spike_amps = np.squeeze(-np.concatenate(amps)).astype("float64")
            spike_depths_array = np.concatenate(spike_depths)
            quality_metrics_df = pd.concat(quality_metrics)

        np.save(output_folder / "spikes.clusters.npy", spike_clusters)
        np.save(output_folder / "spikes.times.npy", spike_times)
        np.save(output_folder / "spikes.amps.npy", spike_amps)
        np.save(output_folder / "spikes.depths.npy", spike_depths_array)
        np.save(
            output_folder / "clusters.peakToTrough.npy",
            cluster_peak_to_trough,
        )
        np.save(
            output_folder / "clusters.channels.npy",
            cluster_channels,
        )
        assert len(spike_clusters) == len(shank_indices)
        assert len(cluster_channels) == len(unit_shank_indices)
        np.save(output_folder / "spike_shank_indices.npy", shank_indices)
        np.save(
            output_folder / "unit_shank_indices.npy",
            unit_shank_indices,
        )

        min_num_channels_waveforms = min(
            [w.shape[1] for w in cluster_waveforms]
        )
        waveforms = [
            w[:, :min_num_channels_waveforms] for w in cluster_waveforms
        ]
        np.save(
            output_folder / "clusters.waveforms.npy",
            np.array(waveforms),
        )
        quality_metrics_df.to_csv(output_folder / "clusters.metrics.csv")