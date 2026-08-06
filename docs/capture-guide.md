# Capture guide

## Recommended camera setup

- Mount or brace the camera so it does not move.
- Keep the complete person, including hands and feet, inside frame.
- Record at 1080p or higher when finger articulation matters.
- Record at 30 or 60 FPS with a short shutter time if possible.
- Leave visual separation between overlapping limbs.

For push-ups, place the camera around hip height and use a side or shallow three-quarter angle. A pure front view creates more left/right overlap and depth ambiguity.

## Existing internet footage

A T-pose or known body measurement is not required. KineTrace estimates one robust skeleton from the visible motion in the complete clip. When choosing among available videos, prefer a fixed camera, limited cuts, a complete body, and an elevated three-quarter view where the two wrists do not overlap. Trim title cards and unrelated shots before processing so they do not influence automatic calibration.

## Hand limitations

Full-body framing makes hands small. When the hand model loses a hand, KineTrace interpolates short bounded gaps and labels those points as inferred. A hand identified as a planted support keeps one reliable articulated pose during contact. Longer gaps remain missing rather than creating arbitrary finger motion.

## Privacy

KineTrace does not upload videos. Source media and generated data are stored under the local `.kinetrace/` directory.
