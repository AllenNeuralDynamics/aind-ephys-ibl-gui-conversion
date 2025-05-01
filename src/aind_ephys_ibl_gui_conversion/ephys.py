from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from scipy import signal
from spikeinterface.core import BaseRecording, get_random_data_chunks
from spikeinterface.core.job_tools import (
    ChunkRecordingExecutor,
    fix_job_kwargs,
)
from spikeinterface.exporters import export_to_phy

def extract_spikes(
    sorting_folder, results_folder, min_duration_secs: int = 300
):
    """
    Extract spike data from a sorting folder and save the results in the specified results folder.

    Parameters
    ----------
    sorting_folder : str
        The path to the folder containing the sorted spike data. This folder is expected to
        contain files or directories related to spike sorting results (e.g., .npy, .csv, etc.).

    results_folder : str
        The path to the folder where the extracted spike data will be saved. The extracted data
        will be written to this folder in an appropriate format.

    min_duration_secs : int, optional, default=300
        The minimum duration (in seconds) of spike events to be considered for extraction.
        Only spike events that last at least this long will be processed. The default value is
        300 seconds (5 minutes).

    Returns
    -------
    None
        This function does not return any value. The extracted spike data is saved directly to
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

    sorting_curated_folder = sorting_folder / "sorting_precurated"
    postprocessed_folder = sorting_folder / "postprocessed"

    # extract stream names

    stream_names, stream_ids = se.get_neo_streams(
        "openephysbinary", ecephys_folder
    )

    neuropix_streams = [s for s in stream_names if "Neuropix" in s]
    probe_names = [s.split(".")[1].split("-")[0] for s in neuropix_streams]

    RMS_WIN_LENGTH_SECS = 3
    WELCH_WIN_LENGTH_SAMPLES = 1024

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
                    postprocessed_folder
                    / f"experiment1_{stream_name}_recording1_group{shank_index}.zarr"
                )

                if analyzer_folder.is_dir():
                    analyzer = si.load_sorting_analyzer(analyzer_folder)
                else:
                    analyzer = si.load_sorting_analyzer_or_waveforms(
                        postprocessed_folder
                        / f"experiment1_{stream_name}_recording1_group{shank_index}"
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
                analyzer = si.load_sorting_analyzer_or_waveforms(
                    postprocessed_folder
                    / f"experiment1_{stream_name}_recording1"
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

        cluster_offset = 0
        peak_channel_offset = 0  # IBL gui uses cluster channels to index for multishank so think this is needed
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

        # for concatenating in case of different number of channels for multiple analyzers
        min_num_channels_waveforms = min(
            [w.shape[1] for w in cluster_waveforms]
        )
        waveforms = [
            w[:, :min_num_channels_waveforms] for w in cluster_waveforms
        ]
        np.save(output_folder / "clusters.waveforms.npy", np.array(waveforms))
        quality_metrics_df.to_csv(output_folder / "clusters.metrics.csv")


def compute_rms(
    recording: BaseRecording,
    verbose: bool = False,
    **job_kwargs,
):
    """
    Compute the RMS of a recording in chunks.

    Parameters
    ----------
    recording: BaseRecording
        The recording object to compute the RMS for.
    {}
    """
    job_kwargs = fix_job_kwargs(job_kwargs)

    # use executor (loop or workers)
    func = _compute_rms_chunk
    init_func = _init_rms_worker
    init_args = (recording,)
    executor = ChunkRecordingExecutor(
        recording,
        func,
        init_func,
        init_args,
        job_name="compute_rms",
        verbose=verbose,
        handle_returns=True,
        **job_kwargs,
    )
    results = executor.run()

    rms_values = np.zeros((len(results), recording.get_num_channels()))
    rms_times = np.zeros((len(results)))

    for i, result in enumerate(results):
        rms_values[i, :], rms_times[i] = result

    return rms_values, rms_times


def _init_rms_worker(recording):
    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["times"] = recording.get_times()
    return worker_ctx


def _compute_rms_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    recording = worker_ctx["recording"]
    times = worker_ctx["times"]

    traces = recording.get_traces(
        start_frame=start_frame,
        end_frame=end_frame,
        segment_index=segment_index,
    )
    rms = np.sqrt(np.mean(traces**2, axis=0))
    # get the middle time of the chunk
    if end_frame < recording.get_num_samples() - 1:
        middle_frame = (start_frame + end_frame) // 2
    else:
        # if we are at the end of the recording, use the middle point of the last chunk
        middle_frame = (start_frame + recording.get_num_samples() - 1) // 2
    # get the time of the middle frame
    rms_time = times[middle_frame]

    return rms, rms_time


def _save_continous_metrics(
    recording: si.BaseRecording,
    output_folder: Path,
    RMS_WIN_LENGTH_SECS=3,
    WELCH_WIN_LENGTH_SAMPLES=2048,
    psd_chunk_duration_sec: float = 1,
    num_chunks: int = 100,
    is_lfp: bool = False,
    tag: Union[str, None] = None,
    **job_kwargs,
):
    """
    Save continuous metrics (e.g., RMS, Welch power spectrum) for the specified channels of a recording.

    Parameters
    ----------
    recording : si.BaseRecording
        A `BaseRecording` object containing the raw data from the recording session. This object is
        expected to have methods to access the data for specific channels (e.g., `get_traces()`).

    output_folder : Path
        The folder where the calculated metrics will be saved. The metrics will be written to files
        in this directory, with filenames based on the `tag` (if provided).

    RMS_WIN_LENGTH_SECS : int or float, optional, default=3
        The length of the window (in seconds) for computing the RMS (Root Mean Square) metric. This
        is the sliding window used to calculate the RMS value for the signal on each channel.

    WELCH_WIN_LENGTH_SAMPLES : int, optional, default=2048
        The length of the window (in samples) used in the Welch method for computing the power spectrum.
        This determines the resolution of the frequency spectrum.

    psd_chunk_duration_s: float, default: 1
        The chunk duration in seconds for the spectral density calculation (on the LFP data).

    num_chunks : int, optional, default=100
        The number of chunks to use for the spectral density calculation (on the LFP data).

    is_lfp : bool, optional, default=False
        If `True`, the function assumes the recording contains local field potentials (LFP). If `False`,
        it assumes the recording contains spikes or another type of signal.

    tag : str or None, optional, default=None
        An optional tag used to distinguish different outputs. If provided, this string will be included
        in the filenames for the saved metrics.

    Returns
    -------
    None
        This function does not return any value. The metrics are saved to the `output_folder` specified
        by the user.
    """
    job_kwargs_ = job_kwargs.copy()
    job_kwargs_["chunk_duration"] = f"{RMS_WIN_LENGTH_SECS}s"

    if is_lfp:
        if tag is not None:
            alf_object_time = f"ephysTimeRmsLF{tag}"
        else:
            alf_object_time = f"ephysTimeRmsLF"
    else:
        if tag is not None:
            alf_object_time = f"ephysTimeRmsAP{tag}"
        else:
            alf_object_time = f"ephysTimeRmsAP"

    rms_data, rms_times = compute_rms(
        recording=recording, verbose=True, **job_kwargs_
    )
    np.save(output_folder / f"_iblqc_{alf_object_time}.rms.npy", rms_data)
    np.save(
        output_folder / f"_iblqc_{alf_object_time}.timestamps.npy", rms_times
    )

    if is_lfp:
        lfp_sample_data = get_random_data_chunks(
            recording,
            num_chunks_per_segment=num_chunks,
            chunk_duration=f"{psd_chunk_duration_sec}s",
            return_scaled=True,
            concatenated=True,
        )
        psd = np.zeros(
            (WELCH_WIN_LENGTH_SAMPLES // 2 + 1, lfp_sample_data.shape[1]),
            dtype=np.float32,
        )
        for i_channel in range(lfp_sample_data.shape[1]):
            freqs, Pxx = signal.welch(
                lfp_sample_data[:, i_channel],
                fs=recording.sampling_frequency,
                nperseg=WELCH_WIN_LENGTH_SAMPLES,
            )
            psd[:, i_channel] = Pxx

        if tag is not None:
            lfp_spectral_file = f'ephysSpectralDensityLF{tag}'
        else:
            lfp_spectral_file = f'ephysSpectralDensityLF'

        np.save(output_folder / f"_iblqc_{lfp_spectral_file}.power.npy", psd)
        np.save(
            output_folder / f"{lfp_spectral_file}.freqs.npy", freqs
        )


def remove_overlapping_channels(recordings) -> list[si.BaseRecording]:
    """
    Remove recordings with overlapping channels from a list of `BaseRecording` objects.

    This function iterates over a list of recordings and identifies recordings with channels
    that overlap with those in other recordings. It returns a list of recordings with no overlapping
    channels.

    Parameters
    ----------
    recordings : list of si.BaseRecording
        A list of `BaseRecording` objects, each representing a recording session. These objects
        should contain methods to retrieve channel information (e.g., `get_channel_ids()`).

    Returns
    -------
    list of si.BaseRecording
        A list of `BaseRecording` objects that do not have any overlapping channels.

    Raises
    ------
    ValueError
        If any of the `BaseRecording` objects in the `recordings` list does not contain valid
        channel information or if the list is empty.

    Notes
    -----
    - The function assumes that the `BaseRecording` objects contain a method `get_channel_ids()`
      that returns a list of channel identifiers for each recording.
    - The function compares the channel identifiers across all recordings to identify overlaps.
    - The order of recordings in the returned list is the same as in the input list, excluding those
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
            channel_id_remove = [
                channel_id
                for channel_id in recording.channel_ids
                if str(index + 1) in channel_id
            ][0]
            channel_ids_to_remove.append(channel_id_remove)

        removed_recordings.append(
            recording.remove_channels(channel_ids_to_remove)
        )

    return removed_recordings


def get_ecephys_stream_names(base_folder: Path) -> tuple[list[str], Path, int]:
    """
    Retrieve the names of available ecephys data streams, along with the associated data directory
    and the number of streams found within the specified folder.

    This function scans a given base folder for available ecephys data streams and returns:
    1. A list of stream names (as strings),
    2. The path to the folder where the data streams are located,
    3. The total number of streams found.

    Parameters
    ----------
    base_folder : Path
        The path to the base folder that contains ecephys data streams. The folder is expected to
        contain subdirectories or files representing the streams.

    Returns
    -------
    tuple of (list of str, Path, int)
        - A list of strings containing the names of the ecephys data streams found in the base folder.
        - The path to the base folder where the streams were located.
        - An integer representing the total number of streams found in the base folder.

    Raises
    ------
    FileNotFoundError
        If the `base_folder` does not exist or is not accessible.

    ValueError
        If no ecephys data streams are found in the `base_folder`.

    Notes
    -----
    - The function assumes that the `base_folder` contains subdirectories or files that can be
      identified as ecephys data streams.
    - The list of stream names may correspond to experimental data streams or other related datasets.
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
    probe_names = [s.split(".")[1].split("-")[0] for s in neuropix_streams]

    return neuropix_streams, ecephys_compressed_folder, num_blocks


def get_mappings(
    main_recordings: dict,
    recording_mappings: dict,
    neuropix_streams: list,
    num_blocks: int,
    ecephys_compressed_folder: Path,
    min_duration_secs: int = 300,
) -> tuple[dict, dict]:
    """
    Generate mappings for the ecephys data streams and their corresponding recording blocks.

    This function takes in the details of the main recordings and their mappings, along with information
    about the neuropix streams, to generate two mappings: one for the data streams and another for the
    associated blocks. The mappings are returned as dictionaries.

    Parameters
    ----------
    main_recordings : dict
        A dictionary where keys represent unique identifiers for recordings and values are
        metadata or objects associated with those recordings. This can include details about the
        recording setup, time, and related information.

    recording_mappings : dict
        A dictionary containing mappings of recording that are short in duration, i.e. surface recording

    neuropix_streams : list of str
        A list of stream names or identifiers for the neuropix data streams. These streams typically
        correspond to the raw or processed data associated with the ecephys recordings.

    num_blocks : int
        The total number of blocks to consider when generating the mappings. This will typically
        correspond to chunks or sections of the recordings that are processed or analyzed separately.

    ecephys_compressed_folder : Path
        The path to the folder where compressed ecephys data is stored. This folder may contain data
        in a format that needs to be uncompressed or processed for further use.

    min_duration_secs : int, optional, default=300
        The minimum duration (in seconds) that a recording must have in order to be included in the
        mapping process. This can be useful to filter out short-duration recordings that are not
        relevant for further analysis.

    Returns
    -------
    tuple of (dict, dict)
        - A dictionary representing the mapping of ecephys data streams to their respective
          recordings and blocks.
        - A second dictionary mapping recording identifiers to specific block details or additional
          metadata.

    Raises
    ------
    ValueError
        If there is an inconsistency between the `main_recordings` and `recording_mappings`, such
        as missing or mismatched data.

    FileNotFoundError
        If the `ecephys_compressed_folder` does not exist or cannot be accessed.

    KeyError
        If a required key is missing in any of the dictionaries (`main_recordings`, `recording_mappings`).

    Notes
    -----
    - The function assumes that the `main_recordings` and `recording_mappings` dictionaries are properly
      structured and contain relevant information for generating the mappings.
    - The `min_duration_secs` parameter helps exclude recordings that are too short to be of interest
      for further analysis.
    - The returned mappings can be used for efficiently organizing and accessing specific parts of
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

        # MULTI SHANKS: groups = np.unique(recording.get_channel_groups()), recording.split_by('group'), {group: channels on shank}
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


def extract_continuous(
    sorting_folder: Path,
    results_folder: Path,
    min_duration_secs: int = 300,
    probe_surface_finding: Union[Path, None] = None,
    use_lfp_cmr: bool = False,
):
    """
    Extract continuous data from sorted recordings and save the results to the specified folder.

    This function processes the sorted data in the provided `sorting_folder` and extracts continuous
    signals, such as local field potentials (LFP) or continuous neural recordings, to be saved in
    the `results_folder`. The extracted data can be filtered by a minimum duration and may use a
    probe surface finding file for additional processing.

    Parameters
    ----------
    sorting_folder : Path
        The path to the folder containing the sorted data. This folder is expected to contain
        spike-sorted recordings, such as `.npy` or `.csv` files, depending on the sorting method used.

    results_folder : Path
        The path to the folder where the processed continuous data will be saved. The extracted signals
        will be written to files within this directory, typically in formats suitable for further analysis.

    min_duration_secs : int, optional, default=300
        The minimum duration (in seconds) of the continuous data that will be included in the extraction.
        Recordings shorter than this duration will be ignored.

    probe_surface_finding : Path or None, optional, default=None
        The path to a file that contains information about the probe surface finding, if applicable.
        This can be used for further processing or filtering of the data based on probe configuration.
        If not provided, no surface finding data will be used.

    use_lfp_cmr : bool, optional, default=False
        If `True`, the function will use the local field potential (LFP) continuous metric results
        (CMR) for additional analysis. If `False`, skips this

    Returns
    -------
    None
        This function does not return any value. The processed continuous data is saved to the
        `results_folder` specified by the user.

    Raises
    ------
    FileNotFoundError
        If the `sorting_folder` or `results_folder` do not exist or cannot be accessed.

    ValueError
        If the `min_duration_secs` is negative or invalid, or if there are issues with the
        probe surface finding file.

    Notes
    -----
    - The function assumes that the `sorting_folder` contains valid sorted data.
    - The extracted continuous data will be saved in a format suitable for further analysis,
      depending on the type of data processed (e.g., LFP, spike data).
    - If `probe_surface_finding` is provided, it must be in a compatible format for additional processing.
    - The `use_lfp_cmr` option should be set to `True` if the analysis involves local field potentials
      and associated metrics.
    """

    session_folder = Path(str(sorting_folder).split("_sorted")[0])

    # At some point the directory structure changed- handle different cases.
    neuropix_streams, ecephys_compressed_folder, num_blocks = (
        get_ecephys_stream_names(session_folder)
    )
    neuropix_streams_surface = (
        []
    )  # try to account for if surface recording is a seperate asset, identified by probe_surface_finding

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
            out_channel_mask = (
                channel_labels == "out"
            )  # TODO: might not work, or adjust threshold, load preprocessed recording

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
        main_recording_lfp = spre.highpass_filter(
            [
                recording
                for recording in main_recordings[stream_name]
                if recording.get_num_samples() == max_samples_lfp
            ][0]
        )

        max_samples_ap = max(
            [
                recording.get_num_samples()
                for recording in main_recordings[
                    stream_name.replace("LFP", "AP")
                ]
            ]
        )
        main_recording_ap = spre.highpass_filter(
            [
                recording
                for recording in main_recordings[
                    stream_name.replace("LFP", "AP")
                ]
                if recording.get_num_samples() == max_samples_ap
            ][0]
        )
        # good_channel_mask = np.isin(recording.channel_ids, analyzer.channel_ids)
        channel_inds = np.arange(recording_ap.get_num_channels())

        print(f"Stream sample rate: {recording_ap.sampling_frequency}")

        _save_continous_metrics(recording_ap, output_folder, channel_inds)
        _save_continous_metrics(
            recording_lfp, output_folder, channel_inds, is_lfp=True
        )

        # save for longer main recording
        _save_continous_metrics(
            main_recording_lfp,
            output_folder,
            channel_inds=np.arange(main_recording_lfp.get_num_channels()),
            TOTAL_SECS=600,
            is_lfp=True,
            tag="Main",
        )
        _save_continous_metrics(
            main_recording_ap,
            output_folder,
            channel_inds=np.arange(main_recording_ap.get_num_channels()),
            TOTAL_SECS=600,
            tag="Main",
        )

        # need appended channel locations so app can show surface recording locations also
        np.save(
            output_folder / "channels.localCoordinates.npy",
            recording_ap.get_channel_locations(),
        )
        np.save(output_folder / "channels.rawInd.npy", channel_inds)
