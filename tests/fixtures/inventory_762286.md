# Retained multisegment regression metadata

Raw asset: `ecephys_762286_2025-01-23_15-37-17`
Code Ocean ID: `a6678a23-1490-4a7b-b0d0-0fa9aa633d99`

Sorted asset: `ecephys_762286_2025-01-23_15-37-17_sorted_2025-07-18_01-56-40`
Code Ocean ID: `3fa5e43a-627f-446f-803c-2514cf6e4f0f`

This is the session used to validate the original largest-segment fix in
conversion PR 42 and issue 36. The compressed fixture contains JSON metadata
only: acquisition headers, AP/LFP segment shapes, probe geometry and legacy
identity, and reduced analyzer attributes. It contains no signals or spikes.

The main AP segment has 93,133,656 samples; the second has 10,227,540.
Both have independently sorted analyzers. Probe identity is stored under
`probes_info`, with unprefixed single-shank contact IDs. The analyzers retain
383 of the raw recording's 384 channels. Main AP and LFP durations agree;
the shorter segment has a duration mismatch and must not invalidate the
explicitly selected largest segment used by the previous converter.
