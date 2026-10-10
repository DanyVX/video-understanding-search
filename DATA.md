# Data registry and permissions

No video, audio, thumbnails, transcripts or benchmark datasets are committed.

For each future clip, add a registry record with:
`video_id`, public source URL or "private consented", license identifier,
permitted uses, attribution, duration, resolution, downloaded checksum,
collection date and split. **Never commit private consent documentation.**

| Video ID | Origin | License | Duration | Resolution | Attribution |
| --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD | TBD |

Allowed: self-recorded media with participant consent, verifiably CC/permissive
footage, public domain or an authorized benchmark subset. Verify dataset
terms on the provider page; MSR-VTT, ActivityNet Captions and QVHighlights are
**not** assumed redistributable. Never commit large binaries, personal media
or datasets by default. Prefer checksum-verified fetch scripts once selected.

Input retention defaults to **none** after indexing, except temporary
processing data required for a bounded job. Until encryption, deletion and
privacy design are implemented, restrict experiments to non-sensitive clips.
