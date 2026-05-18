# Security Policy

## Supported versions

Security fixes are applied to the latest release tag. Older versions are not patched — please update to the current release before reporting an issue against an older binary.

## Reporting a vulnerability

If you discover a security issue, please **do not open a public GitHub issue**. Instead, report it privately:

1. Open a [GitHub Security Advisory](https://github.com/b451c/reamix.me/security/advisories/new) on the repository.
2. Include reproduction steps, affected versions, and any proof-of-concept (audio file, REAPER project file, dylib in question).
3. Expect an initial acknowledgment within 7 days.

## Scope

reamix.me is a REAPER extension that loads as a `.dylib` / `.dll` / `.so` inside the REAPER process. The relevant attack surface includes:

- **File parsing**: WAV / AIFF / FLAC / MP3 / OGG decoding via JUCE's `AudioFormatManager`. Memory-safety bugs in JUCE codecs are upstream; please report those to the JUCE project directly.
- **ONNX model loading**: beat-detection model is downloaded once at first run from a GitHub Release URL pinned in source. Tampering with the URL would require a malicious commit to this repo.
- **REAPER project state**: reamix.me writes timeline edits via REAPER's public API (`SetMediaItemInfo_Value`, `Undo_BeginBlock`, etc.). It does not write outside the user's REAPER project.
- **Network**: the only network call is the one-time model download. No telemetry, no usage tracking, no remote logging.

## Out of scope

- REAPER itself (report to [Cockos](https://forum.cockos.com/forum.php)).
- ONNX Runtime (report to the [ONNX Runtime project](https://github.com/microsoft/onnxruntime/security)).
- JUCE (report to the [JUCE project](https://github.com/juce-framework/JUCE)).
- Operating-system Gatekeeper / SmartScreen / equivalent — these are platform concerns, not reamix-specific.

## Disclosure timeline

We aim for **30 days** between confirmed report and public disclosure. Critical vulnerabilities (remote code execution, data exfiltration) get expedited handling.
