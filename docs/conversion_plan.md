# Conversion decisions use the discovery inventory

`conversion_plan.build_conversion_plan()` applies selection policy to an
inventory without opening recordings or sorting analyzers. The plan records
its input inventory hash, policy, overrides, candidate durations, selected
recording and analyzer family, physical shank coverage, QC inputs, and findings.
The plan contract is version `1.0.0`, independently of inventory and package
versions.

The fan-out worker passes the saved plan and its inventory to both extractors.
`prepare_extraction()` validates the decision and binds its asset-relative
references to mounted roots. Plans retain `execution_enabled=false` because
creating a decision performs no extraction. The worker records activation and
outcome separately in `ephys_extraction.json` inside the ephys collection.

## Main-recording policy

The default chooses the longest recorded block for a collection within its
sorted asset's declared raw sources. It measures duration as the sum of stored
segment sample counts divided by sampling frequency. This excludes unrecorded
wall-clock gaps. The policy is recorded as `longest_recorded_duration` and
`sum_segment_samples_over_rate`.

Selection happens before examining analyzer availability. A long block with
missing sorting stays the selected main block. A shorter recording cannot
silently replace it or supply a missing main shank.

Unknown or nonpositive durations and exact ties leave selection blocked. A
tie uses an absolute tolerance of 1 ns solely for floating-point comparison.
Distinct full stream identities sharing one collection token require an
explicit main override. Storage order and experiment numbering never break
ties. The default requires declared raw source assets; duration does not
establish which source asset produced an ambiguous sorting.

```python
from aind_ephys_ibl_gui_conversion.conversion_plan import build_conversion_plan

plan = build_conversion_plan(inventory, "ProbeA", "sorted-asset-name")
```

`main_recording_id` can explicitly select an inventoried block in scope. Its
duration must still be known. `sorting_family_id` can select an analyzer family
listed in `sorting_families`. A family override does not implicitly change the
main recording; selecting another block requires a main override as well.
Overrides cannot waive conflicting source metadata.

## Sorting families and physical coverage

One family consists of analyzers sharing the sorted asset, experiment, full
stream, and acquisition recording identifier. Group labels stay local to that
family. Unknown identities are not pooled. If several families can supply the
chosen block, the planner lists them and waits for an explicit choice.

All groups in the chosen family are considered together. Mixing grouped and
ungrouped representations, duplicate group labels, overlapping physical
contacts, source disagreements, and acquisition recording conflicts are
reported. Contact comparison uses probe serial, contact identity, coordinates,
and units. Physical shanks come from contact prefixes or recorded probe shank
IDs, never from enumeration of group files.

`shank_coverage` compares recorded and sorted physical shanks in the main block.
Missing shanks make the plan incomplete. The planner does not infer why outputs
are absent or merge independently sorted experiments to fill the gap. Unit IDs
remain local to each selected analyzer; extraction must provide the eventual
output index mapping.

Multiple raw segments retain their original indices and local time basis.
The planner records `main_segment` when every selected analyzer has one segment
and its sample count identifies the same unique raw segment. This check operates
within the source already identified by asset, experiment, full stream, rate,
and physical contacts. The acquisition recording identifier must match one
header with compatible sampling and channels; multiple headers also require the
header and raw-segment counts to agree. The matching header ID is retained as
`acquisition_header_id`. Sample count alone never waives source checks.

The segment decision records raw and analyzer segment indices, sample count,
duration, and evidence. By default it selects the longest raw segment before inspecting available
sortings and chooses its uniquely matching family. A shorter family cannot
replace missing sorting for that segment. An explicit `sorting_family_id` can
select a uniquely mapped shorter segment within the chosen raw recording.
Repeated raw lengths, disagreement among analyzer groups, and genuinely
multisegment analyzers remain unresolved. No session offsets are fabricated.

## Continuous inputs

The plan lists the collection's primary AP/wideband blocks and checks physical
probe identity, contact geometry, sample rate, and duration. Channel banks and
shanks may vary across compatible blocks. Each input retains its own stream,
segments, channel configuration, selection flag, and exclusion reason.

The chosen block gets role `main`, with duration or override evidence. Other
blocks retain role `unknown`. Only an explicitly supplied surface asset assigns
the `surface` role; short duration alone never does. NP1 LFP inputs are paired
with AP blocks within the same source asset and experiment, with compatible
segment durations. LFP blocks do not compete as main AP recordings.

## Findings and readiness

- `ready`: the metadata supports the recorded decisions.
- `incomplete`: some required identity, mapping, geometry, metadata, or shank
  coverage remains unresolved.
- `blocked`: no unambiguous selection exists, an override is invalid, or
  selected inputs conflict.

Readiness does not verify signal contents, units' quality, or the existing
extractor's ability to execute the plan. Inventory hashes bind saved artifacts;
they do not revalidate external storage contents.

In the retained 858398 case, duration selects ProbeA experiment1 (5,634.26 s).
Its two analyzers cover physical shanks 0 and 1, leaving 2 and 3 absent. The
plan remains incomplete and keeps the ten shorter blocks' roles unknown.
The split ProbeB-2 stream instead has a complete main family on physical
shank 1; its collection-local display index does not replace that identity.

## Extraction contract

`prepare_extraction(plan, scoped_inventory, asset_roots, results_folder)` opens
only selected compressed recording paths and the chosen analyzer family. It
checks the plan version and inventory hash, source references, live segment
lengths, sample rates, channel counts, probe serials, physical contacts,
coordinates, channel order, and analyzer unit counts. Paths must remain inside
the explicitly bound source root, including after resolving symlinks. Neither
extractor enumerates streams with Neo or chooses another analyzer when this
context is present. A failed planned extraction never falls back to discovery.

Ready plans execute. An incomplete plan may execute only when its sole finding
is `main_shanks_without_sorting`: available main-family analyzers produce units,
raw channels retain QC coverage, and the missing physical shanks stay in the
saved decision. Other incomplete or blocked plans fail before ALF array output.
A multisegment main recording can execute when `main_segment` identifies its
single sorted segment. The worker checks that saved decision against its
inventory and live analyzer samples. Multisegment analyzers remain unsupported.
Other segments of the main raw recording are excluded with reason
`not_selected_main_segment`, preserving the previous largest-segment QC domain.
Secondary recordings retain separate QC blocks and segment-local clocks.

The selected main segment comes first, followed by secondary recordings
sorted by inventory ID and segment index. Blocks shorter than one analysis window are
excluded with their IDs and reasons recorded in the execution receipt. A main
block shorter than the window fails; another block never takes its place.
Both consumers use the resulting canonical channel table. An explicit main
override remains main even when another QC block is longer. Unknown roles stay
unknown, and a request for surface-only coherence fails without explicit surface
blocks.

Units receive unique dense output rows across analyzers. `clusters.metrics.csv`
retains `source_unit_id`, `analyzer_id`, and `analyzer_group` alongside the dense
`cluster_id`. Spike times use the selected segment's sample-zero clock, without adding the
durations of preceding raw segments. A stable
time sort applies the same permutation to cluster labels, amplitudes, depths,
and physical shank IDs. `clusters.channels.npy` maps template peaks onto the
canonical channel table, rather than storing group-local channel indices.

Planned `clusters.waveforms.npy` stores each unit's peak waveform with a single
channel axis. `clusters.waveformChannels.npy` identifies that channel's table
row. Differing template windows align at sample zero with NaN padding;
`clusters.waveformTimes.npy` supplies their shared time axis in seconds.
Legacy calls without an execution context retain their existing output behavior.

The worker writes an execution receipt with `started`, `complete`, or `failed`
status, the saved plan, actual block/segment membership, and short-window
exclusions. Diagnostics are retained before preflight, so a validation failure
still leaves its input plan and inventory available. A successful extraction
with missing sorting shanks retains the plan's incomplete coverage status.

Legacy probe identity stored in `annotations.probes_info` is normalized for
both inventory and live validation. Conflicting explicit and legacy serials
remain unknown. Standard unprefixed `e<number>` contacts identify physical
shank zero only when explicit shank IDs are absent; group labels do not supply
physical identity. The retained 762286 fixture covers the original largest-
segment regression with two analyzer families and one removed channel.
