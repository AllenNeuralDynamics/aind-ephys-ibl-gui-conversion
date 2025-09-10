"""
Functions to process ephys data
"""

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

import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from .utils import WindowGenerator, fscale, hp, rms

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

# ChatGPT Helper functions

def _sanitize_name(s: str, maxlen: int = 64) -> str:
    """Make a filesystem-safe name using only [A-Za-z0-9._-+] and trim length."""
    out = []
    for ch in str(s):
        if ch.isalnum():
            out.append(ch)
        elif ch in "._-+":
            out.append(ch)
        else:
            out.append("_")
    out = "".join(out).strip("._-")
    out = out[:maxlen]
    return out or "rec"


def _recording_fingerprint(recording: si.BaseRecording, name_hint: str = "") -> str:
    """
    Build a short, stable-ish cache name for a recording.

    Uses: stream name hint, (#ch, fs, #samples), and channel id samples -> sha1[:8].
    """
    try:
        n_ch = int(recording.get_num_channels())
        fs = float(recording.sampling_frequency)
        n_samp = int(recording.get_num_samples())
        ch_ids = list(map(str, recording.channel_ids))
    except Exception:
        n_ch, fs, n_samp, ch_ids = -1, -1.0, -1, []

    payload = {
        "hint": str(name_hint),
        "n_ch": n_ch,
        "fs": fs,
        "n_samp": n_samp,
        "ch_ids_head": ch_ids[:8],
        "ch_ids_tail": ch_ids[-8:] if ch_ids else [],
    }
    h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    base = "{}_{}ch_{}Hz_{}n_{}".format(
        _sanitize_name(name_hint or "rec"),
        n_ch, int(fs), n_samp, h[:8]
    )
    return base


def _cache_or_load_binary(recording: si.BaseRecording, cache_dir: Union[str, Path]) -> si.BaseRecording:
    """
    Cache a recording once to SpikeInterface 'binary_folder' (fast memmap int16).
    If cache exists, load it; otherwise create it.

    Compatible with SI 0.9x–0.100+:
      - tries si.load_extractor on existing folder
      - falls back to si.save(..., format='binary_folder')
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Already cached?
    meta_jsons = ("recording.json", "spikeinterface.json")
    if any((cache_dir / m).exists() for m in meta_jsons):
        try:
            return si.load_extractor(cache_dir)
        except Exception:
            pass  # re-save below if load fails

    # Save once (fast path)
    rec_cached = si.save(recording, folder=str(cache_dir), format="binary_folder")
    return rec_cached

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
    probe_names = [s.split(".")[1].split("_")[0] for s in neuropix_streams]

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


def _save_continous_metrics(
    recording: si.BaseRecording,
    output_folder: Path,
    channel_inds: np.ndarray,
    RMS_WIN_LENGTH_SECS: Union[int, float] = 3,
    WELCH_WIN_LENGTH_SAMPLES: int = 2048,
    TOTAL_SECS: int = 100,
    is_lfp: bool = False,
    tag: Optional[str] = None,
) -> None:
    """
    Save continuous metrics (RMS over time and Welch PSD) for selected channels.

    Performance:
      - assumes 'recording' is an uncompressed cache (binary_folder)
      - does unscaled reads (int16) and casts once to float32
    """
    fs = float(recording.sampling_frequency)

    rms_win_length_samples = int(2 ** np.ceil(np.log2(fs * RMS_WIN_LENGTH_SECS)))
    total_samples = int(min(fs * TOTAL_SECS, recording.get_num_samples()))

    wingen = WindowGenerator(ns=total_samples, nswin=rms_win_length_samples, overlap=0)

    fscale_arr = fscale(WELCH_WIN_LENGTH_SAMPLES, 1.0 / fs, one_sided=True)
    tscale_arr = wingen.tscale(fs=fs)
    n_ch = int(recording.get_num_channels())

    TRMS = np.zeros((wingen.nwin, n_ch), dtype=np.float32)
    nsamples = np.zeros((wingen.nwin,), dtype=np.int32)
    spectral_density = np.zeros((len(fscale_arr), n_ch), dtype=np.float32)

    with tqdm(total=wingen.nwin, desc="Continuous metrics") as pbar:
        for first, last in wingen.firstlast:
            X = recording.get_traces(start_frame=first, end_frame=last, return_scaled=False)  # (samples, ch)
            D = X.T.astype(np.float32, copy=False)  # (ch, samples)

            # high-pass <1 Hz removal (uses your helper)
            D = hp(D, 1.0 / fs, [0, 1])

            iw = wingen.iw
            TRMS[iw, :] = rms(D)
            nsamples[iw] = D.shape[1]

            if (last - first) >= WELCH_WIN_LENGTH_SAMPLES:
                _, w = signal.welch(
                    D,
                    fs=fs,
                    window="hann",
                    nperseg=WELCH_WIN_LENGTH_SAMPLES,
                    detrend="constant",
                    return_onesided=True,
                    scaling="density",
                    axis=-1,
                )
                spectral_density += w.T.astype(np.float32, copy=False)

            pbar.update(1)

    TRMS = TRMS[:, channel_inds]
    spectral_density = spectral_density[:, channel_inds]

    if is_lfp:
        alf_object_time = ("ephysTimeRmsLF" + (tag or "")) if tag else "ephysTimeRmsLF"
        alf_object_freq = ("ephysSpectralDensityLF" + (tag or "")) if tag else "ephysSpectralDensityLF"
    else:
        alf_object_time = ("ephysTimeRmsAP" + (tag or "")) if tag else "ephysTimeRmsAP"
        alf_object_freq = ("ephysSpectralDensityAP" + (tag or "")) if tag else "ephysSpectralDensityAP"

    tdict = {"rms": TRMS.astype(np.single), "timestamps": tscale_arr.astype(np.single)}
    alfio.save_object_npy(output_folder, object=alf_object_time, dico=tdict, namespace="iblqc")

    fdict = {"power": spectral_density.astype(np.single), "freqs": fscale_arr.astype(np.single)}
    alfio.save_object_npy(output_folder, object=alf_object_freq, dico=fdict, namespace="iblqc")
    
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

def _save_channel_metadata(recording: si.BaseRecording, output_folder: Path) -> None:
    """
    Save common channel-level metadata expected by IBL/phy-style tools, using
    what SpikeInterface exposes on the Recording.

    Writes (if available):
      - channels.localCoordinates : Nx2 or Nx3 float32
      - channels.localCoordinate  : same as above (alias for legacy consumers)
      - channels.rawInd           : N int32 (zero-based channel ids)
      - channels.shankIds         : N int16 (if 'group'/'shank' property exists)
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # 1) local coordinates
    locs = None
    try:
        locs = recording.get_channel_locations()  # often Nx2 (x,y) or Nx3 (x,y,z)
    except Exception:
        pass

    if locs is not None:
        locs = np.asarray(locs)
        # canonical IBL name:
        alfio.save_object_npy(
            output_folder, object="channels",
            dico={"localCoordinates": locs.astype(np.float32)},
            namespace="ibl"
        )
        # legacy alias (some code looks for singular):
        alfio.save_object_npy(
            output_folder, object="channels",
            dico={"localCoordinate": locs.astype(np.float32)},
            namespace="ibl"
        )

    # 2) raw channel indices
    try:
        raw_ind = np.asarray(recording.channel_ids)
    except Exception:
        raw_ind = None
    if raw_ind is not None:
        # ensure numeric zero-based
        if raw_ind.dtype.kind not in ("i", "u"):
            # try to coerce to int
            raw_ind = np.asarray([int(x) for x in raw_ind], dtype=np.int32)
        alfio.save_object_npy(
            output_folder, object="channels",
            dico={"rawInd": raw_ind.astype(np.int32)},
            namespace="ibl"
        )

    # 3) shank / group ids
    shank = None
    for key in ("group", "shank", "shank_ids", "shank_id"):
        try:
            vals = recording.get_channel_property(key)
            if vals is not None:
                shank = np.asarray(vals)
                break
        except Exception:
            continue
    if shank is not None:
        # coerce to small int
        if shank.dtype.kind not in ("i", "u"):
            shank = np.asarray([int(x) for x in shank], dtype=np.int16)
        alfio.save_object_npy(
            output_folder, object="channels",
            dico={"shankIds": shank.astype(np.int16)},
            namespace="ibl"
        )

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


def extract_continuous(
    recordings_by_name: Mapping[str, si.BaseRecording],
    results_folder: Union[str, Path],
    total_secs: int = 100,
    rms_win_length_secs: Union[int, float] = 3,
    welch_win_length_samples: int = 2048,
    is_lfp_predicate: Optional[callable] = None,
    channel_selection: Optional[Mapping[str, Sequence[int]]] = None,
    cache_root: Union[str, Path] = "/scratch",
    tag_by_name: Optional[Mapping[str, str]] = None,
    save_channels_metadata: bool = True,
) -> None:
    """
    Compute & save continuous metrics (RMS time series + Welch PSD) for 1+ recordings,
    with per-recording caching under /scratch by default, and restored channels.* saves.

    recordings_by_name : {stream_name -> Recording}
    results_folder     : outputs under <results_folder>/<stream_name>/
    """
    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    if is_lfp_predicate is None:
        def _default_is_lfp(name: str) -> bool:
            return "lfp" in str(name).lower()
        is_lfp_predicate = _default_is_lfp

    for stream_name, rec in recordings_by_name.items():
        # 1) per-recording cache under /scratch/<auto-name>/
        auto_name = _recording_fingerprint(rec, name_hint=stream_name)
        cache_dir = Path(cache_root) / auto_name
        rec_cached = _cache_or_load_binary(rec, cache_dir)

        # 2) per-stream output directory
        out_dir = results_folder / _sanitize_name(stream_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 3) optional channel selection
        if channel_selection is not None and stream_name in channel_selection:
            ch_inds = np.asarray(list(channel_selection[stream_name]), dtype=int)
        else:
            ch_inds = np.arange(rec_cached.get_num_channels(), dtype=int)

        # 4) optional tag for ALF object names
        tag = None
        if tag_by_name is not None and stream_name in tag_by_name:
            tag = str(tag_by_name[stream_name])

        # 5) metrics
        _save_continous_metrics(
            recording=rec_cached,
            output_folder=out_dir,
            channel_inds=ch_inds,
            RMS_WIN_LENGTH_SECS=rms_win_length_secs,
            WELCH_WIN_LENGTH_SAMPLES=welch_win_length_samples,
            TOTAL_SECS=total_secs,
            is_lfp=bool(is_lfp_predicate(stream_name)),
            tag=tag,
        )

        # 6) channel metadata (restored)
        if save_channels_metadata:
            _save_channel_metadata(rec_cached, out_dir)
