# MetaHuman and Blender workflow

KineTrace uses MetaHuman as the presentation character for GEM-X motion. The MediaPipe mannequin is deliberately unchanged. The first production path targets Blender rendering; a browser-optimized MetaHuman is a later, separate asset pipeline.

## Pinned starting point

- Unreal Engine and MetaHuman Creator 5.8
- Blender 4.5 LTS
- Poly Hammer Character DNA 0.12.9 or newer (free core edition)
- Poly Hammer Character Control Rig with Rigify (separately sold convenience layer)

Character DNA imports the MetaHuman head/body DNA and evaluates its RigLogic and body RBF correctives in Blender. Character Control Rig adds animator-friendly controls and recognizes SOMA animation as an import template. MetaHuman and Character DNA's core edition are free under their respective terms; Character Control Rig is not. KineTrace's native NPZ export remains useful if that retargeting layer is replaced later.

## Character preparation

1. Create one athletic, skin-covered MetaHuman in MetaHuman Creator. Use fitted workout clothing and short hair or hair cards for the first test; neither an ecorche surface nor painted muscle regions are part of the target design.
2. Use MetaHuman Creator's DCC Export to produce `head.dna`, `body.dna`, textures, and the package metadata.
3. Keep the exported package outside Git. DNA, textures, generated Blender scenes, and rendered media are local production assets unless their distribution has been reviewed separately.
4. In Blender, install Character DNA and import the DCC package. Confirm that body RBF correctives evaluate before introducing KineTrace motion.
5. Install Character Control Rig, enable Blender's bundled Rigify add-on, generate the metarig, inspect its finger/toe/spine placement, and generate the final control rig.

## Importing KineTrace motion

1. Process the source clip with GEM-X.
2. Download **MetaHuman motion** from the completed job. The file is named `gemx-soma-motion.npz`.
3. In Character Control Rig's Animation panel, choose **Body FK Animation > Import** and select the NPZ.
4. Confirm that the source template is detected as **SOMA**. Set **Source Frame Rate** to the FPS shown by KineTrace; NPZ importers may otherwise assume 30 FPS.
5. Keep **Match Frame Rate** enabled and import onto the generated control rig.
6. Inspect contacts and extreme poses, then make only the necessary IK/control-rig corrections before rendering.

The NPZ contains GEM-X's native axis-angle rotations for all 77 SOMA deform joints plus the grounded root translation. It is more faithful than KineTrace's generic BVH, whose rotations are reconstructed from the normalized landmark contract. KineTrace's mesh-space palm/finger contact correction cannot be represented by the untouched native rotations, so finish planted-hand cleanup on the MetaHuman control rig.

## First acceptance clip

Use one short push-up or squat clip before building clothing and lighting variants. Accept the pipeline only when:

- shoulders and hips retain believable volume through the full range;
- elbows, knees, wrists, and fingers do not collapse or twist;
- planted hands or feet remain close enough to the floor for a short IK cleanup;
- the MetaHuman body correctives remain active during playback and render;
- the result reads as a polished fitness character rather than an anatomy model.

Hair, clothing simulation, muscle-emphasis correctives, final skin shading, and browser optimization come after this motion/deformation test passes.

## References

- [MetaHuman 5.8 release notes](https://dev.epicgames.com/documentation/metahuman/metahuman-5-8-release-notes-in-unreal-engine)
- [MetaHuman Creator DCC Export](https://dev.epicgames.com/documentation/metahuman/metahuman-creator-export-tool-in-unreal-engine)
- [MetaHuman license](https://www.metahuman.com/license)
- [Character DNA](https://github.com/poly-hammer/character-dna-addon)
- [Character Control Rig: SOMA animation import](https://docs.polyhammer.com/character-control-rig-addon/import-animation/)
