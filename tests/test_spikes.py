"""Tests for spikes module."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numcodecs
import numcodecs.pickles
import numpy as np
import pandas as pd
import pytest
import spikeinterface as si
import zarr

from aind_ephys_ibl_gui_conversion import spikes
from aind_ephys_ibl_gui_conversion.spikes import (
    _find_analyzer_folder,
    _quality_metrics_with_labels,
)


def _make_sorting(properties=None, unit_ids=(1, 2)):
    """Create real SI property arrays without recording dependencies."""
    sorting = si.NumpySorting.from_unit_dict(
        {unit: np.array([10 + i, 30 + i]) for i, unit in enumerate(unit_ids)},
        sampling_frequency=30000,
    )
    for key, values in (properties or {}).items():
        sorting.set_property(key, np.asarray(values))
    return sorting


def _make_analyzer(sorting):
    """Provide deterministic converter extensions around a real sorting."""
    metrics = pd.DataFrame({"snr": [4.0, 5.0]}, index=sorting.unit_ids)
    n_spikes = sorting.to_spike_vector().size
    locations = np.zeros(n_spikes, dtype=[("y", "float64")])
    locations["y"] = 100.0
    templates = np.zeros((2, 8, 2))
    templates[:, 3, 0] = -10
    templates[:, 5, 0] = 5
    extensions = {
        "quality_metrics": SimpleNamespace(get_data=lambda: metrics),
        "spike_amplitudes": SimpleNamespace(
            get_data=lambda: np.ones(n_spikes)
        ),
        "spike_locations": SimpleNamespace(get_data=lambda: locations),
        "templates": SimpleNamespace(get_templates=lambda: templates),
    }
    return SimpleNamespace(
        sorting=sorting,
        unit_ids=sorting.unit_ids,
        sampling_frequency=30000,
        get_total_duration=lambda: 1000,
        get_extension=extensions.__getitem__,
    )


def test_existing_labels_do_not_load_curated(tmp_path, monkeypatch):
    """Multi-unit properties win even when a curated directory exists."""
    sorting = _make_sorting(
        {
            "default_qc": [False, True],
            "decoder_label": ["noise", "good"],
            "unitrefine_label": ["ignored", "ignored"],
        }
    )
    analyzer = _make_analyzer(sorting)
    load = Mock(side_effect=AssertionError("Unexpected curated load"))
    monkeypatch.setattr(spikes.si, "load", load)
    result = _quality_metrics_with_labels(analyzer, tmp_path)
    assert result["default_qc"].tolist() == [False, True]
    assert result["unitrefine_label"].tolist() == ["noise", "good"]
    assert list(analyzer.get_extension("quality_metrics").get_data()) == [
        "snr"
    ]
    load.assert_not_called()


def test_missing_curated_keeps_available_analyzer_labels(tmp_path):
    """A missing optional folder leaves unknown labels absent."""
    analyzer = _make_analyzer(
        _make_sorting({"unitrefine_label": ["good", "noise"]})
    )
    result = _quality_metrics_with_labels(analyzer, tmp_path / "missing")
    assert result["unitrefine_label"].tolist() == ["good", "noise"]
    assert "default_qc" not in result


def test_curated_fills_only_missing_labels(tmp_path, monkeypatch):
    """Curated labels fill gaps, align by unit ID, and preserve false QC."""
    analyzer = _make_analyzer(_make_sorting({"default_qc": [False, False]}))
    original = analyzer.get_extension("quality_metrics").get_data()
    original.sort_index(ascending=False, inplace=True)
    curated = _make_sorting(
        {"default_qc": [True, True], "decoder_label": ["good", "noise"]}
    )
    monkeypatch.setattr(spikes.si, "load", Mock(return_value=curated))
    result = _quality_metrics_with_labels(analyzer, tmp_path)
    assert result["default_qc"].tolist() == [False, False]
    assert result["unitrefine_label"].tolist() == ["noise", "good"]
    assert result["cluster_id.1"].tolist() == [2, 1]


@pytest.mark.parametrize("unit_ids", [(2, 1), (1, 3)])
def test_curated_unit_mismatch_is_explicit(tmp_path, monkeypatch, unit_ids):
    """Reject reordered or different curated units before copying labels."""
    analyzer = _make_analyzer(_make_sorting())
    curated = _make_sorting(unit_ids=unit_ids)
    monkeypatch.setattr(spikes.si, "load", Mock(return_value=curated))
    with pytest.raises(ValueError, match="unit IDs do not match"):
        _quality_metrics_with_labels(analyzer, tmp_path)


@pytest.mark.parametrize("n_shanks", [1, 2])
@pytest.mark.parametrize("has_curated", [False, True])
def test_extract_spikes_optional_curated(
    tmp_path, monkeypatch, n_shanks, has_curated
):
    """Single/multi-shank extraction saves labels without obsolete loaders."""
    stream = "Record Node 104#Neuropix-PXI-100.ProbeA-AP"
    sorting_folder = tmp_path / "sorted"
    postprocessed = sorting_folder / "postprocessed"
    postprocessed.mkdir(parents=True)
    results = tmp_path / "results"
    results.mkdir()
    analyzers = {}
    for shank in range(n_shanks):
        suffix = f"_group{shank}" if n_shanks > 1 else ""
        name = f"experiment1_{stream}_recording1{suffix}.zarr"
        folder = postprocessed / name
        folder.mkdir()
        analyzers[folder] = _make_analyzer(_make_sorting())
        if has_curated:
            (sorting_folder / "curated" / name).mkdir(parents=True)
    curated = _make_sorting(
        {"default_qc": [False, True], "decoder_label": ["noise", "good"]}
    )
    load = Mock(return_value=curated)
    monkeypatch.setattr(spikes.si, "load", load)
    monkeypatch.setattr(spikes, "_load_analyzer", analyzers.__getitem__)
    monkeypatch.setattr(
        spikes.se, "get_neo_streams", lambda *args: ([stream], [0])
    )
    spikes.extract_spikes(
        sorting_folder, results, session_folder=tmp_path / "raw"
    )
    output = results / "ProbeA"
    metrics = pd.read_csv(output / "clusters.metrics.csv")
    assert len(metrics) == n_shanks * 2
    assert load.call_count == (n_shanks if has_curated else 0)
    if has_curated:
        assert metrics["default_qc"].tolist() == [False, True] * n_shanks
        assert (
            metrics["unitrefine_label"].tolist()
            == ["noise", "good"] * n_shanks
        )
    else:
        assert "default_qc" not in metrics
        assert "unitrefine_label" not in metrics
    assert np.load(output / "spikes.times.npy").size == n_shanks * 4


class TestFindAnalyzerFolder(unittest.TestCase):
    """Tests for _find_analyzer_folder folder discovery."""

    def setUp(self):
        """Create a temporary postprocessed directory."""
        self._tmp = tempfile.TemporaryDirectory()
        self.postprocessed = Path(self._tmp.name)
        self.stream = "Record Node 104#Neuropix-PXI-100.ProbeA-AP"

    def tearDown(self):
        """Remove the temporary directory."""
        self._tmp.cleanup()

    def _make(self, name):
        """Create an empty folder in the postprocessed directory."""
        folder = self.postprocessed / name
        folder.mkdir()
        return folder

    def test_finds_recording_numbered_zarr(self):
        """A recording-numbered .zarr folder is found (legacy layout)."""
        expected = self._make(f"experiment1_{self.stream}_recording1.zarr")

        result = _find_analyzer_folder(self.postprocessed, self.stream)

        self.assertEqual(result, expected)

    def test_finds_folder_without_recording_number(self):
        """A folder omitting the recording number is found."""
        expected = self._make(f"experiment1_{self.stream}.zarr")

        result = _find_analyzer_folder(self.postprocessed, self.stream)

        self.assertEqual(result, expected)

    def test_finds_non_zarr_recording_folder(self):
        """A non-.zarr (waveforms) folder is found when no .zarr exists."""
        expected = self._make(f"experiment1_{self.stream}_recording1")

        result = _find_analyzer_folder(self.postprocessed, self.stream)

        self.assertEqual(result, expected)

    def test_arbitrary_experiment_and_recording_index(self):
        """Experiment and recording indices other than 1 are matched."""
        expected = self._make(f"experiment3_{self.stream}_recording7.zarr")

        result = _find_analyzer_folder(self.postprocessed, self.stream)

        self.assertEqual(result, expected)

    def test_zarr_preferred_over_non_zarr(self):
        """When both exist, the .zarr folder is returned first."""
        self._make(f"experiment1_{self.stream}_recording1")
        expected = self._make(f"experiment1_{self.stream}_recording1.zarr")

        result = _find_analyzer_folder(self.postprocessed, self.stream)

        self.assertEqual(result, expected)

    def test_returns_none_when_missing(self):
        """None is returned when no matching folder exists."""
        self._make(f"experiment1_{self.stream}_recording1_group0.zarr")

        result = _find_analyzer_folder(self.postprocessed, self.stream)

        self.assertIsNone(result)

    def test_single_shank_ignores_group_folders(self):
        """A shankless lookup must not match a multi-shank group folder."""
        self._make(f"experiment1_{self.stream}_recording1_group0.zarr")
        expected = self._make(f"experiment1_{self.stream}_recording1.zarr")

        result = _find_analyzer_folder(self.postprocessed, self.stream)

        self.assertEqual(result, expected)

    def test_finds_shank_group_folder(self):
        """A group folder is found when shank_index is given."""
        expected = self._make(f"experiment1_{self.stream}_recording1_group1")

        result = _find_analyzer_folder(
            self.postprocessed, self.stream, shank_index=1
        )

        self.assertEqual(result, expected)

    def test_finds_shank_group_without_recording_number(self):
        """A group folder without a recording number is found."""
        expected = self._make(f"experiment1_{self.stream}_group2.zarr")

        result = _find_analyzer_folder(
            self.postprocessed, self.stream, shank_index=2
        )

        self.assertEqual(result, expected)

    def test_shank_lookup_matches_only_requested_group(self):
        """The requested shank index must be selected, not another."""
        self._make(f"experiment1_{self.stream}_recording1_group0.zarr")
        expected = self._make(
            f"experiment1_{self.stream}_recording1_group1.zarr"
        )

        result = _find_analyzer_folder(
            self.postprocessed, self.stream, shank_index=1
        )

        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.skipif(
    not hasattr(numcodecs.pickles, "ALLOW_UNSAFE_PICKLE_DECODE"),
    reason="numcodecs predates the Pickle decode opt-in",
)
def test_load_analyzer_decodes_pickled_extension_data(tmp_path, monkeypatch):
    """Pickled dict extension data loads, and the opt-in does not leak out."""
    monkeypatch.setattr(numcodecs.pickles, "ALLOW_UNSAFE_PICKLE_DECODE", False)
    recording, sorting = si.generate_ground_truth_recording(
        durations=[1.0], num_channels=4, num_units=2, seed=0
    )
    folder = tmp_path / "analyzer.zarr"
    analyzer = si.create_sorting_analyzer(
        sorting, recording, format="zarr", folder=folder, sparse=False
    )
    analyzer.compute("noise_levels")
    # Mirrors how SpikeInterface stores dict-valued extension data.
    group = zarr.open_group(str(folder), mode="a")["extensions/noise_levels"]
    group.create_dataset(
        "params_dict",
        data=np.array([{"key": 1}], dtype=object),
        object_codec=numcodecs.Pickle(),
    )
    group["params_dict"].attrs["dict"] = True
    zarr.consolidate_metadata(str(folder))

    with pytest.raises(numcodecs.errors.UnsafeDecodeError):
        si.load_sorting_analyzer(folder)

    loaded = spikes._load_analyzer(folder)

    ext = loaded.get_extension("noise_levels")
    assert ext.data["params_dict"] == {"key": 1}
    assert numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE is False
