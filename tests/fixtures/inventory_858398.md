# Retained metadata for recording 858398

`inventory_858398.json.gz` contains reduced JSON metadata from the public
`aind-open-data` assets:

- `ecephys_858398_2026-08-04_13-48-57`
- `ecephys_858398_2026-08-04_13-48-57_sorted_2026-10-01_02-40-51`

Captured November 17, 2026. The two top-level maps, `raw` and `sorted`, map
asset-relative paths to JSON objects. They retain 11 Open Ephys recording
headers, 54 compressed recordings (43 probe and 11 auxiliary streams), and
40 analyzer metadata records. No signal, spike, extension, or pickle payloads
are included.

The reduction preserves channel/contact geometry, counts, rates, segment
lengths, experiment annotations, source recording references, and unit counts.
Unrelated acquisition annotations, sorting parameters, and extension metadata
were removed. Preserve the real geometry: experiment1 contains four ProbeA
shanks but has analyzers for only two; later experiments select individual
shanks and depth ranges. These observations do not label acquisition purposes
or establish why sorting outputs are absent.
