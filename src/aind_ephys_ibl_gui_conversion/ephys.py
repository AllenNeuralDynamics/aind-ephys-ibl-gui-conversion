"""
Functions to process ephys data
"""

import logging
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

# here we define some constants used for defining if timestamps are ok
# or should be skipped
ACCEPTED_NEGATIVE_DEVIATION_MS = (
    0.2  # we allow for small negative timestamps diff glitches
)
# maximum number of negative timestamps allowed below the accepted deviation
MAX_NUM_NEGATIVE_TIMESTAMPS = 10
ABS_MAX_TIMESTAMPS_DEVIATION_MS = (
    2  # absolute maximum deviation allowed for timestamps (also positive)
)

MAX_NUM_NEGATIVE_TIMESTAMPS = 10
MAX_TIMESTAMPS_DEVIATION_MS = 1


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
    probe_names = [s.split(".")[1].split("-")[0] for s in neuropix_streams]

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

    Raises
    ------
    ValueError
        If any of the `BaseRecording`
        objects in the `recordings` list does
        not contain valid
        channel information or if the list is empty.

    Notes
    -----
    - The function assumes that the `BaseRecording`
      objects contain a method `get_channel_ids()`
      that returns a list of channel identifiers for each recording.
    - The function compares the channel identifiers
      across all recordings to identify overlaps.
    - The order of recordings in the returned
      list is the same as in the input list, excluding those
      that contain overlapping channels.
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

    Raises
    ------
    FileNotFoundError
        If the `base_folder` does not exist or is not accessible.

    ValueError
        If no ecephys data streams are found in the `base_folder`.

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


def _reset_recordings(
    recording: si.BaseRecording, recording_name: str
) -> None:
    """
    Resets the timestamps of the recording if certain conditions are met.

    This function checks the timestamp differences within
    the recording for potential issues.
    If the following conditions are encountered:
    1. The number of negative timestamp differences
       exceeds the threshold (`MAX_NUM_NEGATIVE_TIMESTAMPS`).
    2. The maximum absolute time difference between
       timestamps exceeds the threshold (`ABS_MAX_TIMESTAMPS_DEVIATION_MS`).

    If either condition is true, the recording's
    timestamps are reset, and a message is logged indicating the issue.

    Parameters:
    ----------
    recording : si.BaseRecording
        The recording object containing timestamp data to be checked.

    recording_name : str
        The name of the recording, used for logging purposes.
    """

    # timestamps should be monotonically increasing,
    # but we allow for small glitches
    skip_times = False
    for segment_index in range(recording.get_num_segments()):
        times = recording.get_times(segment_index=segment_index)
        times_diff_ms = np.diff(times) * 1000
        num_negative_times = np.sum(
            times_diff_ms < -ACCEPTED_NEGATIVE_DEVIATION_MS
        )

        if num_negative_times > MAX_NUM_NEGATIVE_TIMESTAMPS:
            logging.info(
                f"\t{recording_name}:\n\t\tSkipping "
                "timestamps for too many negative "
                f"timestamps diffs below {ACCEPTED_NEGATIVE_DEVIATION_MS}: "
                f"{num_negative_times}"
            )
            skip_times = True
            break
        max_time_diff_ms = np.max(np.abs(times_diff_ms))
        if max_time_diff_ms > ABS_MAX_TIMESTAMPS_DEVIATION_MS:
            logging.info(
                f"\t{recording_name}:\n\t\tSkipping timestamps for too "
                f"large time diff deviation: {max_time_diff_ms} ms"
            )
            skip_times = True
            break

    if skip_times:
        recording.reset_times()


def get_mappings(  # noqa: C901
    main_recordings: dict,
    recording_mappings: dict,
    neuropix_streams: list,
    num_blocks: int,
    ecephys_compressed_folder: Path,
    min_duration_secs: int = 300,
) -> tuple[dict, dict]:
    """
    Generate mappings for the ecephys data streams
    and their corresponding recording blocks.

    This function takes in the details of the
    main recordings and their mappings, along with information
    about the neuropix streams, to generate two mappings:
    one for the data streams and another for the
    associated blocks. The mappings are returned as dictionaries.

    Parameters
    ----------
    main_recordings : dict
        A dictionary where keys represent unique
        identifiers for recordings and values are
        metadata or objects associated with those recordings.
        This can include details about the
        recording setup, time, and related information.

    recording_mappings : dict
        A dictionary containing mappings of recording
        that are short in duration, i.e. surface recording

    neuropix_streams : list of str
        A list of stream names or identifiers for
        the neuropix data streams. These streams typically
        correspond to the raw or processed data
        associated with the ecephys recordings.

    num_blocks : int
        The total number of blocks to consider
        when generating the mappings. This will typically
        correspond to chunks or sections of the
        recordings that are processed or analyzed separately.

    ecephys_compressed_folder : Path
        The path to the folder where compressed
        ecephys data is stored. This folder may contain data
        in a format that needs to be uncompressed or
        processed for further use.

    min_duration_secs : int, optional, default=300
        The minimum duration (in seconds) that a recording
        must have in order to be included in the
        mapping process. This can be useful to
        filter out short-duration recordings that are not
        relevant for further analysis.

    Returns
    -------
    tuple of (dict, dict)
        - A dictionary representing the
          mapping of ecephys data streams to their respective
          recordings and blocks.
        - A second dictionary mapping recording
          identifiers to specific block details or additional
          metadata.

    Raises
    ------
    ValueError
        If there is an inconsistency between the
        `main_recordings` and `recording_mappings`, such
        as missing or mismatched data.

    FileNotFoundError
        If the `ecephys_compressed_folder`
        does not exist or cannot be accessed.

    KeyError
        If a required key is missing in any of
        the dictionaries (`main_recordings`, `recording_mappings`).

    Notes
    -----
    - The function assumes that the `main_recordings` and
      `recording_mappings` dictionaries are properly
      structured and contain relevant information for
      generating the mappings.
    - The `min_duration_secs` parameter helps exclude
      recordings that are too short to be of interest
      for further analysis.
    - The returned mappings can be used for
      efficiently organizing and accessing specific parts of
      the ecephys data based on stream and block identifiers.
    """
    for idx, stream_name in enumerate(neuropix_streams):
        has_lfp = False
        if "LFP" in stream_name:
            continue
        elif "AP" in stream_name:
            has_lfp = True
        else:  # 2.0
            has_lfp = True

        for block_index in range(num_blocks):
            recording = si.read_zarr(
                ecephys_compressed_folder
                / f"experiment{block_index + 1}_{stream_name}.zarr"
            )
            recording_groups = recording.split_by("group")
            if "AP" in stream_name:
                stream_name_lfp = stream_name.replace("AP", "LFP")
                recording_lfp = si.read_zarr(
                    ecephys_compressed_folder
                    / f"experiment{block_index + 1}_{stream_name_lfp}.zarr"
                )
                recording_groups_lfp = recording_lfp.split_by("group")
            else:
                recording_groups_lfp = recording_groups

            for group in recording_groups:
                recording_group = recording_groups[group]

                if "AP" not in stream_name and "LFP" not in stream_name:
                    key = f"{stream_name}-AP"
                else:
                    key = stream_name

                _reset_recordings(recording_group, key)
                if recording_group.get_total_duration() < min_duration_secs:
                    if key not in recording_mappings:
                        recording_mappings[key] = [recording_group]
                    else:
                        recording_mappings[key].append(recording_group)
                else:
                    if key not in main_recordings:
                        main_recordings[key] = [recording_group]
                    else:
                        main_recordings[key].append(recording_group)

                if has_lfp:
                    key = key.replace("AP", "LFP")

                    _reset_recordings(recording_groups_lfp[group], key)
                    if (
                        recording_groups_lfp[group].get_total_duration()
                        < min_duration_secs
                    ):
                        if key not in recording_mappings:
                            recording_mappings[key] = [
                                recording_groups_lfp[group]
                            ]
                        else:
                            recording_mappings[key].append(
                                recording_groups_lfp[group]
                            )
                    else:
                        if key not in main_recordings:
                            main_recordings[key] = [
                                recording_groups_lfp[group]
                            ]
                        else:
                            main_recordings[key].append(
                                recording_groups_lfp[group]
                            )

    return main_recordings, recording_mappings


def _save_rms_and_lfp_spectrum(
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
        psd = np.zeros(
            (2**14 // 2 + 1, lfp_sample_data.shape[1]), dtype=np.float32
        )
        for i_channel in range(lfp_sample_data.shape[1]):
            freqs, Pxx = welch(
                lfp_sample_data[:, i_channel],
                fs=recording.sampling_frequency,
                nperseg=2**14,
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


def extract_continuous(  # noqa: C901
    sorting_folder: Path,
    results_folder: Path,
    min_duration_secs: int = 300,
    probe_surface_finding: Union[Path, None] = None,
    lfp_resampling_rate: float = 1000,
    lfp_freq_min: float = 1,
    lfp_freq_max: float = 300,
    use_lfp_cmr: bool = False,
):
    """
    Extract continuous data from sorted recordings
    and save the results to the specified folder.

    This function processes the sorted data in the provided `sorting_folder`
    and extracts continuous
    signals, such as local field potentials (LFP) or
    continuous neural recordings, to be saved in
    the `results_folder`. The extracted data can be
    filtered by a minimum duration and may use a
    probe surface finding file for additional processing.

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

    use_lfp_cmr : bool, optional, default=False
        If `True`, the function will use the local
        field potential (LFP) continuous metric results
        (CMR) for additional analysis. If `False`, skips this

    Returns
    -------
    None
        This function does not return any value.
        The processed continuous data is saved to the
        `results_folder` specified by the user.
    """

    session_folder = Path(str(sorting_folder).split("_sorted")[0])

    # At some point the directory structure changed- handle different cases.
    neuropix_streams, ecephys_compressed_folder, num_blocks = (
        get_ecephys_stream_names(session_folder)
    )
    # recording is a seperate asset,
    # identified by probe_surface_finding
    neuropix_streams_surface = []
    if probe_surface_finding is not None:
        (
            neuropix_streams_surface,
            ecephys_compressed_folder_surface,
            num_blocks,
        ) = get_ecephys_stream_names(probe_surface_finding)

    recording_mappings = {}
    main_recordings = {}

    main_recordings, recording_mappings = get_mappings(
        main_recordings,
        recording_mappings,
        neuropix_streams,
        num_blocks,
        ecephys_compressed_folder,
        min_duration_secs=min_duration_secs,
    )
    if len(neuropix_streams_surface) > 0:
        main_recordings, recording_mappings = get_mappings(
            main_recordings,
            recording_mappings,
            neuropix_streams_surface,
            num_blocks,
            ecephys_compressed_folder_surface,
            min_duration_secs=min_duration_secs,
        )

    for stream_name, main_recordings_streams in main_recordings.items():
        if "LFP" in stream_name:
            continue

        if stream_name in recording_mappings:
            min_samples = min(
                [
                    recording.get_num_samples()
                    for recording in recording_mappings[stream_name]
                ]
            )
            recordings_sliced = [
                recording.frame_slice(start_frame=0, end_frame=min_samples)
                for recording in recording_mappings[stream_name]
            ]
            main_recordings_sliced = [
                main_recording.frame_slice(
                    start_frame=0, end_frame=min_samples
                )
                for main_recording in main_recordings_streams
            ]

            total_recordings = main_recordings_sliced + recordings_sliced
            recordings_removed = remove_overlapping_channels(total_recordings)
            for recording in recordings_removed:
                recording.reset_times()

            recording_ap = si.aggregate_channels(
                recording_list=recordings_removed
            )
        else:
            min_samples = min(
                [
                    main_recording.get_num_samples()
                    for main_recording in main_recordings_streams
                ]
            )
            main_recordings_sliced = [
                main_recording.frame_slice(
                    start_frame=0, end_frame=min_samples
                )
                for main_recording in main_recordings_streams
            ]

            recordings_removed = remove_overlapping_channels(
                main_recordings_sliced
            )
            for recording in recordings_removed:
                recording.reset_times()

            recording_ap = si.aggregate_channels(
                recording_list=recordings_removed
            )

        print(stream_name)

        probe_name = stream_name.split(".")[1].split("-")[0]

        output_folder = Path(results_folder) / probe_name

        if not output_folder.exists():
            output_folder.mkdir()

        if use_lfp_cmr:
            recording_highpass = spre.highpass_filter(recording_ap)
            _, channel_labels = spre.detect_bad_channels(recording_highpass)
            # TODO: might not work, or adjust threshold,
            # load preprocessed recording
            out_channel_mask = channel_labels == "out"

        if stream_name.replace("AP", "LFP") in main_recordings:
            stream_name = stream_name.replace("AP", "LFP")
            if stream_name in recording_mappings:
                min_samples = min(
                    [
                        recording.get_num_samples()
                        for recording in recording_mappings[stream_name]
                    ]
                )
                recordings_sliced = [
                    recording.frame_slice(start_frame=0, end_frame=min_samples)
                    for recording in recording_mappings[stream_name]
                ]
                main_recordings_lfp = [
                    main_recording.frame_slice(
                        start_frame=0, end_frame=min_samples
                    )
                    for main_recording in main_recordings[stream_name]
                ]
                total_recordings = main_recordings_lfp + recordings_sliced

                recordings_removed = remove_overlapping_channels(
                    total_recordings
                )
                for recording in recordings_removed:
                    recording.reset_times()

                recording_lfp = si.aggregate_channels(
                    recording_list=recordings_removed
                )
            else:
                min_samples = min(
                    [
                        recording.get_num_samples()
                        for recording in main_recordings[stream_name]
                    ]
                )
                main_recordings_lfp = [
                    main_recording.frame_slice(
                        start_frame=0, end_frame=min_samples
                    )
                    for main_recording in main_recordings[stream_name]
                ]

                recordings_removed = remove_overlapping_channels(
                    main_recordings_lfp
                )
                for recording in recordings_removed:
                    recording.reset_times()

                recording_lfp = si.aggregate_channels(
                    recording_list=recordings_removed
                )

            if use_lfp_cmr:
                out_channel_ids = recording_lfp.channel_ids[out_channel_mask]
                if len(out_channel_ids) > 0:
                    recording_lfp = spre.common_reference(
                        recording_lfp,
                        reference="global",
                        ref_channel_ids=out_channel_ids.tolist(),
                    )

        max_samples_lfp = max(
            [
                recording.get_num_samples()
                for recording in main_recordings[stream_name]
            ]
        )
        main_recording_lfp = [
            recording
            for recording in main_recordings[stream_name]
            if recording.get_num_samples() == max_samples_lfp
        ][0]

        max_samples_ap = max(
            [
                recording.get_num_samples()
                for recording in main_recordings[
                    stream_name.replace("LFP", "AP")
                ]
            ]
        )
        logging.info("High pass filtering concatenated AP recording")
        recording_ap = spre.highpass_filter(recording_ap)

        logging.info("High pass filtering main AP recording")
        main_recording_ap = spre.highpass_filter(
            [
                recording
                for recording in main_recordings[
                    stream_name.replace("LFP", "AP")
                ]
                if recording.get_num_samples() == max_samples_ap
            ][0]
        )
        channel_inds = np.arange(recording_ap.get_num_channels())

        logging.info(
            f"Stream sample rate AP: {recording_ap.sampling_frequency}"
        )

        logging.info(
            "Low pass filtering LFP concatenated recording "
            f"with min freq {lfp_freq_min} and max freq {lfp_freq_max}"
        )
        recording_lfp_low_pass = spre.bandpass_filter(
            recording_lfp, freq_min=lfp_freq_min, freq_max=lfp_freq_max
        )
        logging.info(
            f"Resampling LFP concatenated recording to {lfp_resampling_rate}"
        )
        recording_lfp = spre.resample(
            recording_lfp_low_pass, resample_rate=lfp_resampling_rate
        )

        logging.info(
            "Low pass filtering LFP main recording "
            f"with min freq {lfp_freq_min} and max freq {lfp_freq_max}"
        )
        main_recording_lfp_low_pass = spre.bandpass_filter(
            main_recording_lfp, freq_min=lfp_freq_min, freq_max=lfp_freq_max
        )
        logging.info(
            f"Resampling LFP main recording to {lfp_resampling_rate}"
        )
        main_recording_lfp = spre.resample(
            main_recording_lfp_low_pass, resample_rate=lfp_resampling_rate
        )

        logging.info("Computing rms on concatenated recording")
        _save_rms_and_lfp_spectrum(
            recording_ap,
            output_folder,
        )
        logging.info("Computing rms on main recording")
        _save_rms_and_lfp_spectrum(
            main_recording_ap, output_folder, tag="Main"
        )

        logging.info(
            "Computing rms and lfp spectrum for LFP stream"
            " on concatenated recording"
        )
        _save_rms_and_lfp_spectrum(recording_lfp, output_folder, is_lfp=True)
        logging.info(
            "Computing rms and lfp spectrum for LFP stream"
            " on main recording"
        )
        _save_rms_and_lfp_spectrum(
            main_recording_lfp, output_folder, is_lfp=True, tag="Main"
        )

        # need appended channel locations
        # so app can show surface recording locations also
        np.save(
            output_folder / "channels.localCoordinates.npy",
            recording_ap.get_channel_locations(),
        )
        np.save(output_folder / "channels.rawInd.npy", channel_inds)
