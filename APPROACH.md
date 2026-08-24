# Approach

## Problem Interpretation

<!-- Restate the problem in your own words. Explicitly name the documented
     interpretations from CLAUDE.md's "Interpretation vs Fact" section --dialogue-text
     as an input, onset-vs-completion -- and why each was made. -->

## Architecture

<!-- Diagram + stage-by-stage rationale. Start from CLAUDE.md's architecture section
     and update it with anything that changed during implementation. -->

## How the Solution Decides Where to Look

<!-- Subtitle fast-path, ASR track, OCR track -- and why dual-track-by-default was
     chosen over conditional short-circuiting. -->

## How the Relevant Frame Is Determined

<!-- Arbiter policy (the deterministic reconciliation rules) and the Refine stage's
     timestamp-to-frame mapping policy, including the ASR-anchor-vs-frame-boundary
     nuance. -->

## How the Text Is Extracted

<!-- ASR (faster-whisper, word-level timestamps) and OCR (candidate-region detector +
     OCR engine) specifics -- including which candidate-region detector was actually
     chosen in Phase 3 and why. -->

## Ambiguity & Uncertainty Handling

<!-- Arbiter thresholds, AmbiguousResult, low-confidence flagging. -->

## Trade-offs & What Was Cut

<!-- Reconcile CLAUDE.md's cutting list / not-building list against what actually
     happened during the build. -->

## Known Limitations
