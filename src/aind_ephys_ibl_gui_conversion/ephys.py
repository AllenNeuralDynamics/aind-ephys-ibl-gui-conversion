import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm


from scipy import signal
import one.alf.io as alfio
from .utils import WindowGenerator, fscale, hp, rms




import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.exporters import export_to_phy
import spikeinterface.preprocessing as spre

def extract_spikes(sorting_folder,results_folder, min_duration_secs: int = 300):  
    
    session_folder = Path(str(sorting_folder).split('_sorted')[0])
    scratch_folder = Path('/scratch')

    # At some point the directory structure changed- handle different cases.
    ecephys_folder = session_folder / "ecephys_clipped"
    if ecephys_folder.is_dir():
        ecephys_compressed_folder = session_folder / 'ecephys_compressed'
    else:
        ecephys_folder = session_folder/'ecephys'/'ecephys_clipped'
        ecephys_compressed_folder = session_folder /'ecephys'/ 'ecephys_compressed'
    print(f'ecephys folder: {ecephys_folder}')
    print(f'ecephys compressed folder: {ecephys_compressed_folder}')

    sorting_curated_folder = sorting_folder / "sorting_precurated"
    postprocessed_folder = sorting_folder / 'postprocessed'

    # extract stream names

    stream_names, stream_ids = se.get_neo_streams("openephysbinary", ecephys_folder)

    neuropix_streams = [s for s in stream_names if 'Neuropix' in s]
    probe_names = [s.split('.')[1].split('-')[0] for s in neuropix_streams]

    RMS_WIN_LENGTH_SECS = 3
    WELCH_WIN_LENGTH_SAMPLES = 1024

    analyzer_mappings = []
    num_shanks = 0
    shank_glob = tuple(postprocessed_folder.glob('*group*'))
    if shank_glob:
        num_shanks = len(shank_glob)

    for idx, stream_name in enumerate(neuropix_streams):

        if '-LFP' in stream_name:
            continue

        print(stream_name)

        probe_name = probe_names[idx]

        output_folder = Path(results_folder) / probe_name

        if not output_folder.is_dir():
            output_folder.mkdir()

        print('Loading sorting analyzer...')
        if num_shanks > 1:
            for shank_index in range(num_shanks):
                analyzer_folder = postprocessed_folder / f'experiment1_{stream_name}_recording1_group{shank_index}.zarr'

                if analyzer_folder.is_dir():
                    analyzer = si.load_sorting_analyzer(analyzer_folder)
                else:
                    analyzer = si.load_sorting_analyzer_or_waveforms(
                        postprocessed_folder / f'experiment1_{stream_name}_recording1_group{shank_index}'
                    )
                
                if analyzer.get_total_duration() < min_duration_secs:
                    continue

                analyzer_mappings.append(analyzer)
        else:
            analyzer_folder = postprocessed_folder / f'experiment1_{stream_name}_recording1.zarr'
            if analyzer_folder.is_dir():
                analyzer = si.load_sorting_analyzer(analyzer_folder)
            else:
                analyzer = si.load_sorting_analyzer_or_waveforms(
                    postprocessed_folder / f'experiment1_{stream_name}_recording1'
                )
            analyzer_mappings.append(analyzer)


        phy_folder = scratch_folder / f"{postprocessed_folder.parent.name}_phy"

        print('Exporting to phy format...')

        spike_depths = []
        clusters = []
        spike_samples = []
        amps = []
        # save templates
        cluster_channels = []
        cluster_peak_to_trough = []
        cluster_waveforms = []

        templates = []
        quality_metrics = []

        cluster_offset = 0
        peak_channel_offset = 0 # IBL gui uses cluster channels to index for multishank so think this is needed 
        for analyzer in analyzer_mappings:
            export_to_phy(analyzer, 
                        output_folder=phy_folder,
                        compute_pc_features=False,
                        remove_if_exists=True,
                        copy_binary=False,
                        dtype = 'int16')

            spike_locations = analyzer.get_extension("spike_locations").get_data()
            template_ext = analyzer.get_extension("templates")
            templates = template_ext.get_templates()

            for unit_idx, unit_id in enumerate(analyzer.unit_ids):
                waveform = templates[unit_idx,:,:]
                peak_channel = np.argmax(np.max(waveform, 0) - np.min(waveform,0))
                peak_waveform = waveform[:,peak_channel]
                peak_to_trough = (np.argmax(peak_waveform) - np.argmin(peak_waveform)) / 30000.
                cluster_channels.append(peak_channel + peak_channel_offset)
                cluster_peak_to_trough.append(peak_to_trough)
                cluster_waveforms.append(waveform)
            
            peak_channel_offset = np.max(cluster_channels) + 1

            print('Converting data...')

            current_clusters = np.load(phy_folder / "spike_clusters.npy")
            current_clusters = current_clusters + cluster_offset
            cluster_offset =  np.max(current_clusters) + 1
            clusters.append(current_clusters)
            
            spike_samples.append(np.load(phy_folder / "spike_times.npy"))
            amps.append(np.load(phy_folder / "amplitudes.npy"))
            spike_depths.append(spike_locations["y"])

            # save quality metrics
            qm = analyzer.get_extension("quality_metrics")

            qm_data = qm.get_data()

            qm_data.index.name = 'cluster_id'
            qm_data['cluster_id.1'] = qm_data.index.values
            quality_metrics.append(qm_data)

        if len(analyzer_mappings) == 1:
            spike_clusters = np.squeeze(clusters[0].astype('uint32'))
            spike_times = np.squeeze(spike_samples[0] / 30000.).astype('float64')
            spike_amps = np.squeeze(-amps[0]).astype('float64')
            spike_depths_array = spike_depths[0]
            quality_metrics_df = quality_metrics[0]
        else:
            spike_clusters = np.squeeze(np.concatenate(clusters).astype('uint32'))
            spike_times = np.squeeze(np.concatenate(spike_samples) / 30000.).astype('float64')
            spike_amps = np.squeeze(-np.concatenate(amps)).astype('float64')
            spike_depths_array = np.concatenate(spike_depths)
            quality_metrics_df = pd.concat(quality_metrics)

        np.save(output_folder / "spikes.clusters.npy", spike_clusters)
        np.save(output_folder / "spikes.times.npy", spike_times)
        np.save(output_folder / "spikes.amps.npy", spike_amps)
        np.save(output_folder / "spikes.depths.npy", spike_depths_array)
        np.save(output_folder / "clusters.peakToTrough.npy", cluster_peak_to_trough)
        np.save(output_folder / "clusters.channels.npy", cluster_channels)

        # for concatenating in case of different number of channels for multiple analyzers
        min_num_channels_waveforms = min([w.shape[1] for w in cluster_waveforms])
        waveforms = [w[:, :min_num_channels_waveforms] for w in cluster_waveforms]
        np.save(output_folder / "clusters.waveforms.npy", np.array(waveforms))
        quality_metrics_df.to_csv(output_folder / 'clusters.metrics.csv')

def _save_continous_metrics(recording: si.BaseRecording, output_folder: Path, channel_inds: np.ndarray,
                        RMS_WIN_LENGTH_SECS = 3,
                       WELCH_WIN_LENGTH_SAMPLES=2048,
                       TOTAL_SECS = 100, is_lfp: bool = False, tag: str | None = None):
    rms_win_length_samples = 2 ** np.ceil(np.log2(recording.sampling_frequency * RMS_WIN_LENGTH_SECS))
    total_samples = int(np.min([recording.sampling_frequency * TOTAL_SECS, recording.get_num_samples()]))

    # the window generator will generates window indices
    wingen = WindowGenerator(ns=total_samples, nswin=rms_win_length_samples, overlap=0)

    win = {'TRMS': np.zeros((wingen.nwin, recording.get_num_channels())),
            'nsamples': np.zeros((wingen.nwin,)),
            'fscale': fscale(WELCH_WIN_LENGTH_SAMPLES, 1 / recording.sampling_frequency, one_sided=True),
            'tscale': wingen.tscale(fs=recording.sampling_frequency)}

    win['spectral_density'] = np.zeros((len(win['fscale']), recording.get_num_channels()))

    with tqdm(total=wingen.nwin) as pbar:

        for first, last in wingen.firstlast:

            D = recording.get_traces(start_frame=first, end_frame=last).T

            # remove low frequency noise below 1 Hz
            D = hp(D, 1 / recording.sampling_frequency, [0, 1])
            iw = wingen.iw
            win['TRMS'][iw, :] = rms(D)
            win['nsamples'][iw] = D.shape[1]

            # the last window may be smaller than what is needed for welch
            if last - first < WELCH_WIN_LENGTH_SAMPLES:
                continue

            # compute a smoothed spectrum using welch method
            _, w = signal.welch(
                D, fs=recording.sampling_frequency, window='hann', nperseg=WELCH_WIN_LENGTH_SAMPLES,
                detrend='constant', return_onesided=True, scaling='density', axis=-1
            )
            win['spectral_density'] += w.T
            # print at least every 20 windows
            if (iw % min(20, max(int(np.floor(wingen.nwin / 75)), 1))) == 0:
                pbar.update(iw)

    win['TRMS'] = win['TRMS'][:,channel_inds]
    win['spectral_density'] = win['spectral_density'][:,channel_inds]

    if is_lfp:
        if tag is not None:
            alf_object_time = f'ephysTimeRmsLF{tag}'
            alf_object_freq = f'ephysSpectralDensityLF{tag}'
        else:
            alf_object_time = f'ephysTimeRmsLF'
            alf_object_freq = f'ephysSpectralDensityLF'
    else:
        if tag is not None:
            alf_object_time = f'ephysTimeRmsAP{tag}'
            alf_object_freq = f'ephysSpectralDensityAP{tag}'
        else:
            alf_object_time = f'ephysTimeRmsAP'
            alf_object_freq = f'ephysSpectralDensityAP'

    tdict = {'rms': win['TRMS'].astype(np.single), 'timestamps': win['tscale'].astype(np.single)}
    alfio.save_object_npy(output_folder, object=alf_object_time, dico=tdict, namespace='iblqc')

    fdict = {'power': win['spectral_density'].astype(np.single),
                'freqs': win['fscale'].astype(np.single)}
    alfio.save_object_npy(
        output_folder, object=alf_object_freq, dico=fdict, namespace='iblqc')

def remove_overlapping_channels(recordings) -> list[si.BaseRecording]:
    removed_recordings = []
    channel_locations_seen = set()
    for index, recording in enumerate(recordings):
        remove_indices = []
        channel_locations = [tuple(location) for location in recording.get_channel_locations()]
        for location in channel_locations:
            location = tuple(location)
            if location not in channel_locations_seen:
                channel_locations_seen.add(location)
            else:
                index = channel_locations.index(location)
                remove_indices.append(index)

        channel_ids_to_remove = [f'CH{idx + 1}' for idx in remove_indices]
        removed_recordings.append(recording.remove_channels(channel_ids_to_remove))
    
    return removed_recordings

def get_ecephys_stream_names(base_folder: Path) -> tuple[list[str], Path, int]:
    # At some point the directory structure changed- handle different cases.
    ecephys_folder = base_folder / "ecephys_clipped"
    if ecephys_folder.is_dir():
        ecephys_compressed_folder = base_folder / 'ecephys_compressed'
    else:
        ecephys_folder = base_folder / 'ecephys' / 'ecephys_clipped'
        ecephys_compressed_folder = base_folder /'ecephys'/ 'ecephys_compressed'
    print(f'ecephys folder: {ecephys_folder}')
    print(f'ecephys compressed folder: {ecephys_compressed_folder}')

    # extract stream names
    stream_names, stream_ids = se.get_neo_streams("openephysbinary", ecephys_folder)
    num_blocks = se.get_neo_num_blocks("openephysbinary", ecephys_folder)

    neuropix_streams = [s for s in stream_names if 'Neuropix' in s]
    probe_names = [s.split('.')[1].split('-')[0] for s in neuropix_streams]

    return neuropix_streams, ecephys_compressed_folder, num_blocks

def get_mappings(main_recordings: dict, recording_mappings: dict, neuropix_streams: list, num_blocks: int,
                 ecephys_compressed_folder: Path, min_duration_secs: int = 300) -> tuple[dict, dict]:
    for idx, stream_name in enumerate(neuropix_streams):
        has_lfp = False
        if 'LFP' in stream_name:
            continue
        elif 'AP' in stream_name:
            has_lfp = True
        else: # 2.0
            has_lfp = True

        # MULTI SHANKS: groups = np.unique(recording.get_channel_groups()), recording.split_by('group'), {group: channels on shank}
        for block_index in range(num_blocks):
            recording = si.read_zarr(ecephys_compressed_folder / f"experiment{block_index + 1}_{stream_name}.zarr")
            recording_groups = recording.split_by('group')
            if 'AP' in stream_name:
                stream_name_lfp = stream_name.replace('AP', 'LFP')
                recording_lfp = si.read_zarr(ecephys_compressed_folder / f"experiment{block_index + 1}_{stream_name_lfp}.zarr")
                recording_groups_lfp = recording_lfp.split_by('group')
            else:
                recording_groups_lfp = recording_groups
            
            for group in recording_groups:
                recording_group = recording_groups[group]

                if 'AP' not in stream_name and 'LFP' not in stream_name:
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
                    key = key.replace('AP', 'LFP')

                    if recording_groups_lfp[group].get_total_duration() < min_duration_secs:
                        if key not in recording_mappings:
                            recording_mappings[key] = [recording_groups_lfp[group]]
                        else:
                            recording_mappings[key].append(recording_groups_lfp[group])
                    else:
                        if key not in main_recordings:
                            main_recordings[key] = [recording_groups_lfp[group]]
                        else:
                            main_recordings[key].append(recording_groups_lfp[group])
    
    return main_recordings, recording_mappings

def extract_continuous(sorting_folder: Path,results_folder: Path, min_duration_secs: int = 300,
                       probe_surface_finding: Path| None = None):

    session_folder = Path(str(sorting_folder).split('_sorted')[0])

    # At some point the directory structure changed- handle different cases.
    neuropix_streams, ecephys_compressed_folder, num_blocks = get_ecephys_stream_names(session_folder)
    neuropix_streams_surface = [] # try to account for if surface recording is a seperate asset, identified by probe_surface_finding

    if probe_surface_finding is not None:
        neuropix_streams_surface, ecephys_compressed_folder_surface, num_blocks = get_ecephys_stream_names(probe_surface_finding)

    recording_mappings = {}
    main_recordings = {}

    main_recordings, recording_mappings = get_mappings(main_recordings, recording_mappings, neuropix_streams, num_blocks, ecephys_compressed_folder,
                                            min_duration_secs=min_duration_secs)
    if len(neuropix_streams_surface) > 0:
        main_recordings, recording_mappings = get_mappings(main_recordings, recording_mappings, neuropix_streams_surface, num_blocks, 
                                                ecephys_compressed_folder_surface, min_duration_secs=min_duration_secs)
    
    for stream_name, main_recordings_streams in main_recordings.items():
        if 'LFP' in stream_name:
            continue
        
        if stream_name in recording_mappings:
            min_samples = min([recording.get_num_samples() for recording in recording_mappings[stream_name]])
            recordings_sliced = [recording.frame_slice(start_frame=0, end_frame=min_samples) for recording in recording_mappings[stream_name]]
            main_recordings_sliced = [main_recording.frame_slice(start_frame=0, end_frame=min_samples) for main_recording in main_recordings_streams]
            
            total_recordings = main_recordings_sliced + recordings_sliced
            recordings_removed = remove_overlapping_channels(total_recordings)
            recording_ap = si.aggregate_channels(recording_list=recordings_removed)
        else:
            min_samples = min([main_recording.get_num_samples() for main_recording in main_recordings_streams])
            main_recordings_sliced = [main_recording.frame_slice(start_frame=0, end_frame=min_samples) for main_recording in main_recordings_streams]

            recordings_removed = remove_overlapping_channels(main_recordings_sliced)
            recording_ap = si.aggregate_channels(recording_list=recordings_removed)

        print(stream_name)

        probe_name = stream_name.split('.')[1].split('-')[0]

        output_folder = Path(results_folder) / probe_name

        if not output_folder.exists():
            output_folder.mkdir()

        recording_highpass = spre.highpass_filter(recording_ap)
        _, channel_labels = spre.detect_bad_channels(recording_highpass)
        out_channel_mask = channel_labels == "out" # TODO: might not work, or adjust threshold, load preprocessed recording

        if stream_name.replace('AP', 'LFP') in main_recordings:
            stream_name = stream_name.replace('AP', 'LFP')
            if stream_name in recording_mappings:
                min_samples = min([recording.get_num_samples() for recording in recording_mappings[stream_name]])
                recordings_sliced = [recording.frame_slice(start_frame=0, end_frame=min_samples) for recording in recording_mappings[stream_name]]
                main_recordings_lfp = [main_recording.frame_slice(start_frame=0, end_frame=min_samples) for main_recording in main_recordings[stream_name]]
                total_recordings = main_recordings_lfp + recordings_sliced

                recordings_removed = remove_overlapping_channels(total_recordings)
                recording_lfp = si.aggregate_channels(recording_list=recordings_removed)
            else:
                min_samples = min([recording.get_num_samples() for recording in main_recordings[stream_name]])
                main_recordings_lfp = [main_recording.frame_slice(start_frame=0, end_frame=min_samples) for main_recording in main_recordings[stream_name]]

                recordings_removed = remove_overlapping_channels(main_recordings_lfp)
                recording_lfp = si.aggregate_channels(recording_list=recordings_removed)

            out_channel_ids = recording_lfp.channel_ids[out_channel_mask]
            if len(out_channel_ids) > 0:
                recording_lfp = spre.common_reference(recording_lfp, reference='global', ref_channel_ids=out_channel_ids.tolist())

        max_samples = max([recording.get_num_samples() for recording in main_recordings[stream_name]])
        main_recording_lfp = spre.highpass_filter([recording for recording in main_recordings[stream_name] if recording.get_num_samples() == max_samples][0])
        main_recording_ap = spre.highpass_filter([recording for recording in main_recordings[stream_name.replace('LFP', 'AP')] if recording.get_num_samples() == max_samples][0])
        #good_channel_mask = np.isin(recording.channel_ids, analyzer.channel_ids)
        channel_inds = np.arange(recording_ap.get_num_channels())

        print(f'Stream sample rate: {recording_ap.sampling_frequency}')

        _save_continous_metrics(recording_ap, output_folder, channel_inds)
        _save_continous_metrics(recording_lfp, output_folder, channel_inds, is_lfp=True)

        # save for longer main recording
        _save_continous_metrics(main_recording_lfp, output_folder, channel_inds=np.arange(main_recording_lfp.get_num_channels()),
                                                         TOTAL_SECS=600, is_lfp=True, tag='Main')
        _save_continous_metrics(main_recording_ap, output_folder, channel_inds=np.arange(main_recording_ap.get_num_channels()),
                                TOTAL_SECS=600, tag='Main')

        # need appended channel locations so app can show surface recording locations also
        np.save(output_folder / 'channels.localCoordinates.npy', recording_ap.get_channel_locations())
        np.save(output_folder / 'channels.rawInd.npy', channel_inds)
