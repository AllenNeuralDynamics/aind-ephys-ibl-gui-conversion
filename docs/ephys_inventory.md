# Electrophysiology discovery inventory

`inventory.build_inventory()` writes observations without selecting conversion
inputs. Inventory version `1.0.0` is separate from the package version and the
GUI datapackage schema. Discovery writes `ephys_inventory.json`; the separate
conversion plan selects inputs for extraction.

The reader uses Open Ephys `structure.oebin`, SpikeInterface Zarr v2 attributes
and array headers, and consolidated or unconsolidated analyzer metadata. Where
supported, it reads five timestamp scalars per segment using bounded byte ranges,
or an explicit segment start time. These observations are stored in each
segment's `timing` field. It never opens voltage chunks, sorting arrays,
extensions, or pickled provenance.
Unsupported or unreadable metadata produces an incomplete source record.

## Version 1 structure

| Field | Contents |
|---|---|
| `inventory_version`, `reader_version` | Artifact contract and generating package version |
| `assets` | Logical source keys, optional Code Ocean IDs/names/aliases, declared source-asset keys, raw/sorted role, metadata completeness |
| `recorded_streams` | Every continuous stream declared in each Open Ephys node/experiment/recording header, with rate and channel count |
| `recordings` | Compressed storage artifacts, declared segment count, segment indices and sample/channel counts, rate, channel-configuration reference |
| `channel_configurations` | Producer probe serial/model, ordered contact IDs/positions, device-channel indices, shank IDs, and units |
| `sortings` | Every analyzer, experiment/stream/recording/group identifiers, retained channels, segment lengths, unit count, channel-configuration reference, source experiment annotation and producer recording reference |
| `relationships` | Candidate header memberships and sorting sources, evidence, and three-valued compatibility checks |
| `findings` | Diagnostic code, source asset and relative metadata path, optional affected record ID and exception type |
| `metadata_complete` | Whether all supplied assets could be inventoried without metadata-reading failures; false for no supplied assets |
| `scope` | Present on worker slices; identifies the collection and sorted-recording scope |

Record IDs hash the logical asset key, relative source path, and record kind.
They survive mount relocation. In Code Ocean, the trigger uses asset IDs as
logical keys. Local discovery uses manifest/source names when IDs are unknown.
Changing the source identity intentionally changes its record IDs.

Local mounts, credentials, and signed URLs are excluded. A sorting's recorded
upstream folder path is retained as descriptive producer evidence, never used
to locate data in the current deployment.

## Relationships and missing information

Path-derived identities are labelled `storage_name`. Sorting candidates are
restricted to declared source assets when supplied, then matched by experiment
and full stream identity. Their experiment annotations, sample counts, rates,
and contact geometry are compared independently. Boolean checks mean agreement
or disagreement; `null` means insufficient evidence. A candidate is not an
authoritative source reference or a conversion selection.

Multiple candidates remain ambiguous. Sample-count disagreement is reported,
not repaired: a future conversion plan must interpret any producer-recorded
slicing, segment selection, or concatenation before using that relationship.
Multiple header recordings remain candidates for a stored artifact; segment
index does not establish which acquisition recording it represents.

The [conversion planner](conversion_plan.md) now records duration-based main
selection, analyzer families, QC inputs, and unresolved mappings separately
from these observations. Both worker extractors consume those saved decisions.

Missing metadata is distinct from an absent stream. A header with no compressed
counterpart is reported. A raw recording with no observed analyzer is reported
without inventing a cause. Missing analyzers do not establish zero units,
failed sorting, or acquisition purpose. Known recordings and analyzers remain
in the inventory even if a sibling record cannot be read.

Physical shanks remain separate from recording-local group labels. Contact
comparison requires probe serial, contact ID, and coordinate units; it does
not normalize groups across experiments. Acquisition roles stay unknown.

Metadata completeness describes reading, not agreement or conversion readiness.
An inventory can be complete and contain conflicting source candidates.
Discovery logs counts and completeness; detailed findings remain in the JSON.

## Access and handoffs

`LocalMetadataStore` resolves an explicitly supplied asset root. Deployment
adapters implement `MetadataStore` for other locations. Reads are capped at
8 MiB per metadata object. The preprocessing S3 adapter lists one directory
level at a time, caches JSON reads, and supports explicit anonymous or signed
access. It never lists recursively through signal chunks.

Preprocessing writes the inventory before filtering manifest rows. Worker
slices contain their sorted asset, declared source recordings, any explicitly
supplied surface asset, and the collection's observations with their referenced
configurations. Each slice accompanies `config.json` through Flatten. No slice
combines sortings or assigns a common time axis.

The retained 858398 metadata fixture exercises changing stream membership,
four-shank and single-shank configurations, auxiliary streams, missing sorting
outputs, and sorting on only two of four simultaneously recorded shanks.
