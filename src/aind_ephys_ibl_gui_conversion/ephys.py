"""
Functions to process ephys data
"""

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from scipy.signal import welch
from spikeinterface.core import get_random_data_chunks
from spikeinterface.exporters import export_to_phy
from spikeinterface.exporters.to_ibl import compute_rms


STREAM_PROBE_REGEX = re.compile(r"^Record Node \d+#[^.]+\.(.+?)(-AP|-LFP)?$")


def _stream_to_probe_name(stream_name: str) -> str | None:
    """
    Extract probe name from Open Ephys stream name.

    Parses stream names from Neuropixels recordings to extract the probe
    identifier, stripping optional -AP (action potential) or -LFP (local field
    potential) suffixes.

    Parameters
    ----------
    stream_name : str
        Open Ephys stream name following the pattern:
        "Record Node {id}#{device}.{probe_name}[-AP|-LFP]"

    Returns
    -------
    str or None
        The extracted probe name (e.g., "ProbeA", "45883-1"), or None if
        the stream name does not match the expected format.

    Examples
    --------
    >>> _stream_to_probe_name("Record Node 104#Neuropix-PXI-100.ProbeA-AP")
    'ProbeA'

    >>> _stream_to_probe_name("Record Node 109#Neuropix-PXI-100.45883-1")
    '45883-1'

    >>> _stream_to_probe_name("InvalidFormat")
    None

    Notes
    -----
    The function uses a regular expression to match the Open Ephys stream
    naming convention. The pattern captures the probe identifier between the
    last period and optional -AP/-LFP suffix.
    """
    m = STREAM_PROBE_REGEX.match(stream_name)
    if m is not None:
        return m.group(1)
    return None


def extract_spikes(  # noqa: C901
    sorting_folder, results_folder, min_duration_secs: int = 300
):
    """
    Extract spike data from a sorting folder and
    save the results in the specified results folder.

    Parameters
    ----------
    sorting_folder : str
        The path to the folder containing the sorted spike data.
        This folder is expected to
        contain files or directories related to
        spike sorting results (e.g., .npy, .csv, etc.).

    results_folder : str
        The path to the folder where the extracted
        spike data will be saved. The extracted data
        will be written to this folder in an appropriate format.

    min_duration_secs : int, optional, default=300
        The minimum duration (in seconds) of spike events
        to be considered for extraction.
        Only spike events that last at least this
        long will be processed. The default value is
        300 seconds (5 minutes).

    Returns
    -------
    None
        This function does not return any value.
        The extracted spike data is saved directly to
        the `results_folder`.
    """

    session_folder = Path(str(sorting_folder).split("_sorted")[0])
    scratch_folder = Path("/scratch")

    # At some point the directory structure changed- handle different cases.
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

    # extract stream names

    stream_names, stream_ids = se.get_neo_streams(
        "openephysbinary", ecephys_folder
    )

    neuropix_streams = [s for s in stream_names if "Neuropix" in s]
    probe_names = [_stream_to_probe_name(s) for s in neuropix_streams]

    for idx, stream_name in enumerate(neuropix_streams):
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
        # save templates
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
            # current_clusters = current_clusters + cluster_offset
            # cluster_offset =  np.max(current_clusters) + 1
            clusters.append(current_clusters)

            for cluster in current_clusters:
                shank_indices.append(index)

            spike_samples.append(np.load(phy_folder / "spike_times.npy"))
            amps.append(np.load(phy_folder / "amplitudes.npy"))
            spike_depths.append(spike_locations["y"])

            # save quality metrics
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
            output_folder / "clusters.peakToTrough.npy", cluster_peak_to_trough
        )
        np.save(output_folder / "clusters.channels.npy", cluster_channels)
        assert len(spike_clusters) == len(shank_indices)
        assert len(cluster_channels) == len(unit_shank_indices)
        np.save(output_folder / "spike_shank_indices.npy", shank_indices)
        np.save(output_folder / "unit_shank_indices.npy", unit_shank_indices)

        # for concatenating in case of
        # different number of channels for multiple analyzers
        min_num_channels_waveforms = min(
            [w.shape[1] for w in cluster_waveforms]
        )
        waveforms = [
            w[:, :min_num_channels_waveforms] for w in cluster_waveforms
        ]
        np.save(output_folder / "clusters.waveforms.npy", np.array(waveforms))
        quality_metrics_df.to_csv(output_folder / "clusters.metrics.csv")


def remove_overlapping_channels(recordings) -> list[si.BaseRecording]:
    """
    Remove recordings with overlapping channels
    from a list of `BaseRecording` objects.

    This function iterates over a list of recordings
    and identifies recordings with channels
    that overlap with those in other recordings.
    It returns a list of recordings with no overlapping
    channels.

    Parameters
    ----------
    recordings : list of si.BaseRecording
        A list of `BaseRecording` objects, each representing
        a recording session. These objects
        should contain methods to retrieve
        channel information (e.g., `get_channel_ids()`).

    Returns
    -------
    list of si.BaseRecording
        A list of `BaseRecording` objects that
        do not have any overlapping channels.
    """

    removed_recordings = []
    channel_locations_seen = set()
    for index, recording in enumerate(recordings):
        remove_indices = []
        channel_locations = [
            tuple(location) for location in recording.get_channel_locations()
        ]
        for location in channel_locations:
            if location not in channel_locations_seen:
                channel_locations_seen.add(location)
            else:
                index = channel_locations.index(location)
                remove_indices.append(index)

        channel_ids_to_remove = []
        for index in remove_indices:
            channel_ids_to_remove.append(recording.channel_ids[index])

        removed_recordings.append(
            recording.remove_channels(channel_ids_to_remove)
        )

    return removed_recordings


def get_ecephys_stream_names(base_folder: Path) -> tuple[list[str], Path, int]:
    """
    Retrieve the names of available ecephys data
    streams, along with the associated data directory
    and the number of streams found within the specified folder.

    This function scans a given base folder for
    available ecephys data streams and returns:
    1. A list of stream names (as strings),
    2. The path to the folder where the data streams are located,
    3. The total number of streams found.

    Parameters
    ----------
    base_folder : Path
        The path to the base folder that
        contains ecephys data streams. The folder is expected to
        contain subdirectories or files representing the streams.

    Returns
    -------
    tuple of (list of str, Path, int)
        - A list of strings containing the names
          of the ecephys data streams found in the base folder.
        - The path to the base folder where the streams were located.
        - An integer representing the total number of streams
          found in the base folder.

    Notes
    -----
    - The function assumes that the `base_folder` contains
      subdirectories or files that can be
      identified as ecephys data streams.
    - The list of stream names may correspond to
      experimental data streams or other related datasets.
    """

    # At some point the directory structure changed- handle different cases.
    ecephys_folder = base_folder / "ecephys_clipped"
    if ecephys_folder.is_dir():
        ecephys_compressed_folder = base_folder / "ecephys_compressed"
    else:
        ecephys_folder = base_folder / "ecephys" / "ecephys_clipped"
        ecephys_compressed_folder = (
            base_folder / "ecephys" / "ecephys_compressed"
        )
    print(f"ecephys folder: {ecephys_folder}")
    print(f"ecephys compressed folder: {ecephys_compressed_folder}")

    # extract stream names
    stream_names, stream_ids = se.get_neo_streams(
        "openephysbinary", ecephys_folder
    )
    num_blocks = se.get_neo_num_blocks("openephysbinary", ecephys_folder)

    neuropix_streams = [s for s in stream_names if "Neuropix" in s]

    return neuropix_streams, ecephys_compressed_folder, num_blocks


def process_lfp_stream(
    recording: si.BaseRecording,
    is_1_0_probe: bool,
    freq_min: float,
    freq_max: float,
    decimation_factor: int,
) -> si.BaseRecording:
    """
    Apply LFP preprocessing to a Neuropixels recording.

    For Neuropixels 1.0 probes, the function applies a bandpass filter
    to the recording between the specified frequency limits.
    For Neuropixels 2.0 probes, it additionally downsamples the filtered
    signal by the given decimation factor (typically to ~1.25 kHz).

    Parameters
    ----------
    recording : si.BaseRecording
        The input recording extractor containing the raw LFP signal.
    is_1_0_probe : bool
        True if the recording originates from a Neuropixels 1.0 probe;
        False if from a Neuropixels 2.0 probe.
    freq_min : float
        The lower cutoff frequency for the bandpass filter (in Hz).
    freq_max : float
        The upper cutoff frequency for the bandpass filter (in Hz).
    decimation_factor : int
        The factor by which to downsample the
        signal (only used for 2.0 probes).

    Returns
    -------
    si.BaseRecording
        The processed LFP recording after
        filtering (and decimation if applicable).
    """
    if is_1_0_probe:  # 1.0 probe, only need to bandpass
        logging.info(
            f"1.0 Probe found. Applying bandpass filter "
            f"with freq min {freq_min} "
            f"and freq max {freq_max}"
        )
        return spre.bandpass_filter(
            recording, freq_min=freq_min, freq_max=freq_max
        )
    else:  # 2.0 probe, decimate to 1250
        logging.info("Found 2.0 probe")
        logging.info(
            f"Applying bandpass filter with freq min {freq_min} "
            f"and freq max {freq_max}"
        )
        recording_lfp_bandpass = spre.bandpass_filter(
            recording, freq_min=freq_min, freq_max=freq_max
        )
        logging.info(f"Applying decimation with factor {decimation_factor}")
        return spre.decimate(
            recording_lfp_bandpass, decimation_factor=decimation_factor
        )


def get_neuropixel_lfp_stream(
    recording: si.BaseRecording,
    stream_name: str,
    ecephys_compressed_folder: Path,
    block_index: int,
) -> tuple[si.BaseRecording, bool]:
    """
    Retrieve the appropriate LFP recording for a given Neuropixels stream.

    For Neuropixels 1.0 probes, the LFP is stored in a separate stream file
    (e.g., replacing "AP" with "LFP" in the stream name). For Neuropixels 2.0
    probes, the LFP is embedded within the same recording stream and does not
    require a separate read.

    Parameters
    ----------
    recording : si.BaseRecording
        The AP or combined recording extractor corresponding to the stream.
    stream_name : str
        The name of the Neuropixels stream (e.g., "probeA-AP" or "probeA-LFP").
    ecephys_compressed_folder : Path
        Path to the folder containing the compressed Zarr files
        for each stream.
    block_index : int
        Index of the experiment block to load (used to build the filename).

    Returns
    -------
    tuple[si.BaseRecording, bool]
        A tuple containing:
        - The LFP recording extractor (`si.BaseRecording`).
        - A boolean flag indicating whether the probe is
          Neuropixels 1.0 (`True`)
          or Neuropixels 2.0 (`False`).
    """
    if "AP" in stream_name:  # 1.0 probe - seperate stream
        stream_name_lfp = stream_name.replace("AP", "LFP")
        recording_group_lfp = si.read_zarr(
            ecephys_compressed_folder
            / f"experiment{block_index + 1}_{stream_name_lfp}.zarr"
        )
        recording_lfp = recording_group_lfp
        is_1_0_probe = True
    else:  # 2.0 probe
        recording_lfp = recording
        is_1_0_probe = False

    return recording_lfp, is_1_0_probe


def get_stream_mappings(
    neuropix_streams: list,
    num_blocks: int,
    ecephys_compressed_folder: Path,
    min_duration_secs: int = 300,
    freq_min: float = 1,
    freq_max: float = 300,
    decimation_factor: int = 24,
) -> tuple[dict, dict, dict, dict]:
    """
    Generate mappings between Neuropixels streams and their corresponding
    AP and LFP recordings, separating main (long) recordings from
    surface-finding (short) recordings.

    This function reads Neuropixels recording data from compressed Zarr files,
    splits them by recording groups, and applies appropriate preprocessing:
    - AP data: high-pass filtered.
    - LFP data: bandpass filtered (and decimated for 2.0 probes).

    Recordings shorter than `min_duration_secs` are assumed to be
    surface-finding sessions, while longer recordings are considered
    main sessions.

    Parameters
    ----------
    neuropix_streams : list
        List of stream names (e.g., ["probeA-AP", "probeB-AP", ...]).
        Streams containing "LFP" are ignored, as LFPs are derived internally.
    num_blocks : int
        Number of experiment blocks to process.
    ecephys_compressed_folder : Path
        Path to the folder containing the compressed Zarr recordings.
    min_duration_secs : int, optional
        Minimum duration (in seconds) separating main from surface recordings.
        Defaults to 300 seconds.
    freq_min : float, optional
        Lower cutoff frequency (Hz) for LFP bandpass filtering.
        Defaults to 1 Hz.
    freq_max : float, optional
        Upper cutoff frequency (Hz) for LFP bandpass filtering.
        Defaults to 300 Hz.
    decimation_factor : int, optional
        Downsampling factor for LFP streams
        (used only for Neuropixels 2.0 probes).
        Defaults to 24.

    Returns
    -------
    tuple[dict, dict, dict, dict]
        A tuple of four dictionaries (each a `defaultdict(list)`):
        - `main_recordings_ap`: High-pass filtered AP recordings
           for main sessions.
        - `surface_recordings_ap`: High-pass filtered AP
           recordings for surface sessions.
        - `main_recordings_lfp`: Processed LFP recordings
           for main sessions.
        - `surface_recordings_lfp`: Processed LFP recordings
          for surface sessions.
    """
    main_recordings_ap = defaultdict(list)
    surface_recordings_ap = defaultdict(list)

    main_recordings_lfp = defaultdict(list)
    surface_recordings_lfp = defaultdict(list)

    for idx, stream_name in enumerate(neuropix_streams):
        if "LFP" in stream_name:
            continue

        for block_index in range(num_blocks):
            recording = si.read_zarr(
                ecephys_compressed_folder
                / f"experiment{block_index + 1}_{stream_name}.zarr"
            )
            recording_groups = recording.split_by("group")

            for group in recording_groups:
                logging.info(
                    f"Processing stream {stream_name} for block "
                    f"{block_index} and group {group}"
                )
                recording_group = recording_groups[group]

                recording_time = (
                    recording_group.get_num_samples()
                    / recording_group.sampling_frequency
                )
                logging.info("Applying high pass filter to AP stream")
                recording_group_ap_highpass = spre.highpass_filter(
                    recording_group
                )

                recording_lfp, is_1_0_probe = get_neuropixel_lfp_stream(
                    recording_group,
                    stream_name,
                    ecephys_compressed_folder,
                    block_index,
                )
                recording_lfp_processed = process_lfp_stream(
                    recording_lfp,
                    is_1_0_probe,
                    freq_min,
                    freq_max,
                    decimation_factor,
                )

                # assume this is a surface finding recording
                if recording_time < min_duration_secs:
                    surface_recordings_ap[stream_name].append(
                        recording_group_ap_highpass
                    )
                    surface_recordings_lfp[stream_name].append(
                        recording_lfp_processed
                    )
                else:
                    main_recordings_ap[stream_name].append(
                        recording_group_ap_highpass
                    )
                    main_recordings_lfp[stream_name].append(
                        recording_lfp_processed
                    )

    return (
        main_recordings_ap,
        surface_recordings_ap,
        main_recordings_lfp,
        surface_recordings_lfp,
    )


def save_rms_and_lfp_spectrum(
    recording: si.BaseRecording,
    output_folder: Path,
    n_jobs: int = 10,
    is_lfp: bool = False,
    tag: Union[str, None] = None,
):
    """
    Saves rms and lfp spectrum for the given recording

    Parameters
    ----------
    recording: si.BaseRecording
        The recording to run correlation on

    output_folder: Path
        The output folder to save outputs to

    n_jobs: int, default = 10
        The number of jobs to parallelize rms

    is_lfp: bool, default = False
        If recording is LFP stream

    tag : str or None, optional, default=None
        An optional tag used to distinguish different outputs.
        If provided, this string will be included
        in the filenames for the saved metrics.
    """
    rms, rms_times = compute_rms(recording, n_jobs=n_jobs)

    if not is_lfp:
        if tag is None:
            np.save(output_folder / "_iblqc_ephysTimeRmsAP.rms.npy", rms)
            np.save(
                output_folder / "_iblqc_ephysTimeRmsAP.timestamps.npy",
                rms_times,
            )
        else:
            np.save(output_folder / f"_iblqc_ephysTimeRmsAP{tag}.rms.npy", rms)
            np.save(
                output_folder / f"_iblqc_ephysTimeRmsAP{tag}.timestamps.npy",
                rms_times,
            )
    else:
        if tag is None:
            np.save(output_folder / "_iblqc_ephysTimeRmsLF.rms.npy", rms)
            np.save(
                output_folder / "_iblqc_ephysTimeRmsLF.timestamps.npy",
                rms_times,
            )
        else:
            np.save(output_folder / f"_iblqc_ephysTimeRmsLF{tag}.rms.npy", rms)
            np.save(
                output_folder / f"_iblqc_ephysTimeRmsLF{tag}.timestamps.npy",
                rms_times,
            )

    if is_lfp:
        lfp_sample_data = get_random_data_chunks(
            recording,
            num_chunks_per_segment=100,
            chunk_duration="1s",
            concatenated=True,
        )
        fs = recording.sampling_frequency

        # Target frequency resolution for PSD (0.5 Hz)
        target_freq_resolution = 0.5
        nperseg = int(fs / target_freq_resolution)
        nperseg = 2 ** int(np.log2(nperseg))  # round to nearest power of 2

        # Preallocate PSD array
        psd = np.zeros(
            (nperseg // 2 + 1, lfp_sample_data.shape[1]), dtype=np.float32
        )

        for i_channel in range(lfp_sample_data.shape[1]):
            freqs, Pxx = welch(
                lfp_sample_data[:, i_channel],
                fs=recording.sampling_frequency,
                nperseg=nperseg,
            )
            psd[:, i_channel] = Pxx

        freqs = freqs.astype(np.float32)
        if tag is None:
            np.save(
                output_folder / "_iblqc_ephysSpectralDensityLF.power.npy", psd
            )
            np.save(
                output_folder / "_iblqc_ephysSpectralDensityLF.freqs.npy",
                freqs,
            )
        else:
            np.save(
                output_folder
                / f"_iblqc_ephysSpectralDensityLF{tag}.power.npy",
                psd,
            )
            np.save(
                output_folder
                / f"_iblqc_ephysSpectralDensityLF{tag}.freqs.npy",
                freqs,
            )


def get_concatenated_recordings(
    main_recordings: list[si.BaseRecording],
    surface_recordings: list[si.BaseRecording],
) -> si.BaseRecording:
    """
    Concatenate main and surface recordings after aligning duration and
    removing overlapping channels.

    This function truncates all recordings to the minimum duration among
    the surface recordings to ensure equal length, removes overlapping
    channels across probes, and aggregates the resulting signals into a
    single combined recording using SpikeInterface utilities.

    Parameters
    ----------
    main_recordings : list of si.BaseRecording
        List of main (long-duration) recordings to include.
    surface_recordings : list of si.BaseRecording
        List of surface-finding (short-duration) recordings used to determine
        the truncation length.

    Returns
    -------
    si.BaseRecording
        A concatenated recording extractor containing all main and surface
        recordings (with overlapping channels removed and durations aligned).
    """
    min_samples = min(
        [recording.get_num_samples() for recording in surface_recordings]
    )
    recordings_sliced = [
        recording.frame_slice(start_frame=0, end_frame=min_samples)
        for recording in surface_recordings
    ]
    main_recordings_sliced = [
        main_recording.frame_slice(start_frame=0, end_frame=min_samples)
        for main_recording in main_recordings
    ]

    total_recordings = main_recordings_sliced + recordings_sliced
    recordings_with_overlapping_channels_removed = remove_overlapping_channels(
        total_recordings
    )

    combined_recordings = si.aggregate_channels(
        recording_list=recordings_with_overlapping_channels_removed
    )
    return combined_recordings


def get_main_recording_from_list(
    recordings: list[si.BaseRecording],
) -> si.BaseRecording:
    """
    Gets the main recording by returning recording
    with largest number of samples

    Parameters
    ----------
    recordings: list[si.BaseRecording]
        The list of recordings

    Returns:
    si.BaseRecording
        The recording with the largest number of samples
    """

    return max(recordings, key=lambda r: r.get_num_samples())


def process_raw_data(
    main_recording: si.BaseRecording,
    recording_combined: Union[si.BaseRecording, None],
    stream_name: str,
    results_folder: str,
    is_lfp: bool,
) -> None:
    """
    Processes raw data for a given stream by computing RMS and (if applicable)
    LFP power spectrum for both the main and
    combined (concatenated) recordings.

    This function saves the resulting metrics to an output folder named
    after the probe associated with the stream.

    Parameters
    ----------
    main_recording : si.BaseRecording
        The main recording object for the stream.

    recording_combined : si.BaseRecording or None
        The concatenated recording that includes both main and surface
        recordings. If None, only the main recording is processed.

    stream_name : str
        The name of the recording stream (e.g., 'imec0.ap').

    results_folder : str
        Path to the base folder where output files will be saved.

    is_lfp : bool
        Whether this is an LFP recording. If True, LFP spectrum analysis
        will also be performed.

    Returns
    -------
    None
        The function saves the computed RMS and LFP spectrum results
        to disk and does not return any value.
    """
    probe_name = _stream_to_probe_name(stream_name)
    output_folder = Path(results_folder) / probe_name
    logging.info(
        f"Creating output directory at {output_folder}" " if it does not exist"
    )
    output_folder.mkdir(exist_ok=True)

    if recording_combined is not None:
        logging.info(
            "Running RMS and LFP spectrum (if LFP recording) "
            f"on concatenated recording for stream {stream_name}"
        )
        save_rms_and_lfp_spectrum(
            recording_combined, output_folder, is_lfp=is_lfp
        )

    logging.info(
        "Running RMS and LFP spectrum (if LFP recording) "
        f"on main recording for stream {stream_name}"
    )
    save_rms_and_lfp_spectrum(
        main_recording, output_folder, is_lfp=is_lfp, tag="Main"
    )


def extract_continuous(
    sorting_folder: Path,
    results_folder: Path,
    min_duration_secs: int = 300,
    probe_surface_finding: Union[Path, None] = None,
    lfp_resampling_rate: float = 1000,
    lfp_freq_min: float = 1,
    lfp_freq_max: float = 300,
    num_parallel_jobs: int = 10,
    decimation_factor: int = 24,
):
    """
    Extract features from raw data
    and save the results to the specified folder.

    Parameters
    ----------
    sorting_folder : Path
        The path to the folder containing the sorted data.
        This folder is expected to contain
        spike-sorted recordings, such as `.npy` or `.csv` files,
        depending on the sorting method used.

    results_folder : Path
        The path to the folder where the processed continuous
        data will be saved. The extracted signals
        will be written to files within this directory,
        typically in formats suitable for further analysis.

    min_duration_secs : int, optional, default=300
        The minimum duration (in seconds) of the continuous data
        that will be included in the extraction.
        Recordings shorter than this duration will be ignored.

    probe_surface_finding : Path or None, optional, default=None
        The path to a file that contains information about
        the probe surface finding, if applicable.
        This can be used for further processing or
        filtering of the data based on probe configuration.
        If not provided, no surface finding data will be used.

    lfp_resampling_rate: float, default = 1000
        The rate to resample the LFP recording

    lfp_freq_min: float, defaut = 1,
        The min cutoff frequency to low pass filter
        LFP recording

    lfp_freq_max: float, default = 300,
        The max cutoff frequency to low pass filter
        LFP recording

    num_parallel_jobs: int, default = 10
        Number of parallel jobs to use

    decimation_factor: int, default = 24
        Decimation factor for downsampling
    """

    session_folder = Path(str(sorting_folder).split("_sorted")[0])

    # At some point the directory structure changed- handle different cases.
    neuropix_streams, ecephys_compressed_folder, num_blocks = (
        get_ecephys_stream_names(session_folder)
    )
    # surface recording is a seperate asset,
    # identified by probe_surface_finding
    neuropix_streams_surface = []
    if probe_surface_finding is not None:
        (
            neuropix_streams_surface,
            ecephys_compressed_folder_surface,
            num_blocks,
        ) = get_ecephys_stream_names(probe_surface_finding)

    (
        main_recordings_ap,
        surface_recordings_ap,
        main_recordings_lfp,
        surface_recordings_lfp,
    ) = get_stream_mappings(
        neuropix_streams,
        num_blocks,
        ecephys_compressed_folder,
        min_duration_secs=min_duration_secs,
        freq_min=lfp_freq_min,
        freq_max=lfp_freq_max,
        decimation_factor=decimation_factor,
    )
    if (
        len(neuropix_streams_surface) > 0
    ):  # a seperate asset has been provided for surface recording
        (
            main_recordings_separate_ap,
            surface_recordings_separate_ap,
            main_recordings_separate_lfp,
            surface_separate_recordings_lfp,
        ) = get_stream_mappings(
            neuropix_streams_surface,
            num_blocks,
            ecephys_compressed_folder_surface,
            min_duration_secs=min_duration_secs,
        )

        # TODO: combine this with mappings above for
        # seperate surface finding asset

    logging.info(
        "Looking at AP recordings, "
        "will concatenate if surface recordings are present"
    )
    for stream_name_ap in main_recordings_ap:
        recording_concatenated_ap = None

        if stream_name_ap in surface_recordings_ap:
            logging.info("Surface AP recordings found, concatenating")
            recording_concatenated_ap = get_concatenated_recordings(
                list(main_recordings_ap.values()),
                list(surface_recordings_ap.values()),
            )

        main_recording_ap = get_main_recording_from_list(
            list(main_recordings_ap.values())
        )
        logging.info("Processing raw AP data - Computing rms")
        process_raw_data(
            main_recording_ap,
            recording_concatenated_ap,
            stream_name_ap,
            results_folder,
            is_lfp=False,
        )

    logging.info(
        "Looking at LFP recordings "
        "will concatenate if surface recordings are present"
    )
    for stream_name_lfp in main_recordings_lfp:
        recording_concatenated_lfp = None

        if stream_name_lfp in surface_recordings_lfp:
            logging.info("Surface LFP recordings found, concatenating")
            recording_concatenated_lfp = get_concatenated_recordings(
                list(main_recordings_lfp.values()),
                list(surface_recordings_lfp.values()),
            )

        main_recording_lfp = get_main_recording_from_list(
            list(main_recordings_lfp.values())
        )
        logging.info(
            "Processing raw LFP data - Computing rms and LFP Spectrum"
        )
        process_raw_data(
            main_recording_lfp,
            recording_concatenated_lfp,
            stream_name_lfp,
            results_folder,
            is_lfp=True,
        )
