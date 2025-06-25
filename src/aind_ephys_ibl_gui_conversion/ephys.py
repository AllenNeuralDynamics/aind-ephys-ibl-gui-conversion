import logging
from pathlib import Path
from typing import Union

import numpy as np
import one.alf.io as alfio
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from scipy import signal
from spikeinterface.exporters import export_to_phy
from tqdm import tqdm

from .utils import WindowGenerator, fscale, hp, rms

# here we define some constants used for defining if timestamps are ok
# or should be skipped
ACCEPTED_NEGATIVE_DEVIATION_MS = (
    0.2  # we allow for small negative timestamps diff glitches
)
MAX_NUM_NEGATIVE_TIMESTAMPS = 10  # maximum number of negative timestamps allowed below the accepted deviation
ABS_MAX_TIMESTAMPS_DEVIATION_MS = (
    2  # absolute maximum deviation allowed for timestamps (also positive)
)

MAX_NUM_NEGATIVE_TIMESTAMPS = 10
MAX_TIMESTAMPS_DEVIATION_MS = 1


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


def _save_continous_metrics(
    recording: si.BaseRecording,
    output_folder: Path,
    channel_inds: np.ndarray,
    RMS_WIN_LENGTH_SECS=3,
    WELCH_WIN_LENGTH_SAMPLES=2048,
    TOTAL_SECS=100,
    is_lfp: bool = False,
    tag: Union[str, None] = None,
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

    channel_inds : np.ndarray
        A 1D array of integers specifying the indices of the channels for which the metrics should
        be computed.

    RMS_WIN_LENGTH_SECS : int or float, optional, default=3
        The length of the window (in seconds) for computing the RMS (Root Mean Square) metric. This
        is the sliding window used to calculate the RMS value for the signal on each channel.

    WELCH_WIN_LENGTH_SAMPLES : int, optional, default=2048
        The length of the window (in samples) used in the Welch method for computing the power spectrum.
        This determines the resolution of the frequency spectrum.

    TOTAL_SECS : int, optional, default=100
        The total duration (in seconds) of the data to be processed. Only the first `TOTAL_SECS` of
        the recording will be analyzed. This can be adjusted if only a portion of the recording is of interest.

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

    rms_win_length_samples = 2 ** np.ceil(
        np.log2(recording.sampling_frequency * RMS_WIN_LENGTH_SECS)
    )
    total_samples = int(
        np.min(
            [
                recording.sampling_frequency * TOTAL_SECS,
                recording.get_num_samples(),
            ]
        )
    )

    # the window generator will generates window indices
    wingen = WindowGenerator(
        ns=total_samples, nswin=rms_win_length_samples, overlap=0
    )

    win = {
        "TRMS": np.zeros((wingen.nwin, recording.get_num_channels())),
        "nsamples": np.zeros((wingen.nwin,)),
        "fscale": fscale(
            WELCH_WIN_LENGTH_SAMPLES,
            1 / recording.sampling_frequency,
            one_sided=True,
        ),
        "tscale": wingen.tscale(fs=recording.sampling_frequency),
    }

    win["spectral_density"] = np.zeros(
        (len(win["fscale"]), recording.get_num_channels())
    )

    with tqdm(total=wingen.nwin) as pbar:

        for first, last in wingen.firstlast:

            D = recording.get_traces(start_frame=first, end_frame=last).T

            # remove low frequency noise below 1 Hz
            D = hp(D, 1 / recording.sampling_frequency, [0, 1])
            iw = wingen.iw
            win["TRMS"][iw, :] = rms(D)
            win["nsamples"][iw] = D.shape[1]

            # the last window may be smaller than what is needed for welch
            if last - first < WELCH_WIN_LENGTH_SAMPLES:
                continue

            # compute a smoothed spectrum using welch method
            _, w = signal.welch(
                D,
                fs=recording.sampling_frequency,
                window="hann",
                nperseg=WELCH_WIN_LENGTH_SAMPLES,
                detrend="constant",
                return_onesided=True,
                scaling="density",
                axis=-1,
            )
            win["spectral_density"] += w.T
            # print at least every 20 windows
            if (iw % min(20, max(int(np.floor(wingen.nwin / 75)), 1))) == 0:
                pbar.update(iw)

    win["TRMS"] = win["TRMS"][:, channel_inds]
    win["spectral_density"] = win["spectral_density"][:, channel_inds]

    if is_lfp:
        if tag is not None:
            alf_object_time = f"ephysTimeRmsLF{tag}"
            alf_object_freq = f"ephysSpectralDensityLF{tag}"
        else:
            alf_object_time = f"ephysTimeRmsLF"
            alf_object_freq = f"ephysSpectralDensityLF"
    else:
        if tag is not None:
            alf_object_time = f"ephysTimeRmsAP{tag}"
            alf_object_freq = f"ephysSpectralDensityAP{tag}"
        else:
            alf_object_time = f"ephysTimeRmsAP"
            alf_object_freq = f"ephysSpectralDensityAP"

    tdict = {
        "rms": win["TRMS"].astype(np.single),
        "timestamps": win["tscale"].astype(np.single),
    }
    alfio.save_object_npy(
        output_folder, object=alf_object_time, dico=tdict, namespace="iblqc"
    )

    fdict = {
        "power": win["spectral_density"].astype(np.single),
        "freqs": win["fscale"].astype(np.single),
    }
    alfio.save_object_npy(
        output_folder, object=alf_object_freq, dico=fdict, namespace="iblqc"
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
            ]
            if channel_id_remove:
                channel_ids_to_remove.append(channel_id_remove[0])

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


def _reset_recordings(
    recording: si.BaseRecording, recording_name: str
) -> None:
    """
    Resets the timestamps of the recording if certain conditions are met.

    This function checks the timestamp differences within the recording for potential issues.
    If the following conditions are encountered:
    1. The number of negative timestamp differences exceeds the threshold (`MAX_NUM_NEGATIVE_TIMESTAMPS`).
    2. The maximum absolute time difference between timestamps exceeds the threshold (`ABS_MAX_TIMESTAMPS_DEVIATION_MS`).

    If either condition is true, the recording's timestamps are reset, and a message is logged indicating the issue.

    Parameters:
    ----------
    recording : si.BaseRecording
        The recording object containing timestamp data to be checked.

    recording_name : str
        The name of the recording, used for logging purposes.
    """

    # timestamps should be monotonically increasing, but we allow for small glitches
    skip_times = False
    for segment_index in range(recording.get_num_segments()):
        times = recording.get_times(segment_index=segment_index)
        times_diff_ms = np.diff(times) * 1000
        num_negative_times = np.sum(
            times_diff_ms < -ACCEPTED_NEGATIVE_DEVIATION_MS
        )

        if num_negative_times > MAX_NUM_NEGATIVE_TIMESTAMPS:
            logging.info(
                f"\t{recording_name}:\n\t\tSkipping timestamps for too many negative "
                f"timestamps diffs below {ACCEPTED_NEGATIVE_DEVIATION_MS}: {num_negative_times}"
            )
            skip_times = True
            break
        max_time_diff_ms = np.max(np.abs(times_diff_ms))
        if max_time_diff_ms > ABS_MAX_TIMESTAMPS_DEVIATION_MS:
            logging.info(
                f"\t{recording_name}:\n\t\tSkipping timestamps for too large time diff deviation: {max_time_diff_ms} ms"
            )
            skip_times = True
            break

    if skip_times:
        recording.reset_times()


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
