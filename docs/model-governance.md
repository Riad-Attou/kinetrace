# Model governance

Generated animation may be used in a commercial downstream project, so model selection is treated as a commercial-compatibility requirement even though KineTrace itself is personal software.

The first backend uses official Google MediaPipe model bundles. Before adding any alternative model, record:

- source repository and exact version;
- code license;
- model-weight terms;
- training-dataset restrictions disclosed by the provider;
- whether commercial use and generated-output use are addressed;
- model checksum.

An open-source implementation license does not automatically determine the terms of its checkpoints or training data. Alternative checkpoints are therefore audited individually before they become a selectable backend.

## NVIDIA GEM-X experimental backend

- Source: `NVlabs/GEM-X`, pinned by `scripts/setup_gemx.sh` to commit `32992550dba114c62243fb55e361311972dce8f9`.
- Source license: Apache-2.0.
- Checkpoints: `nvidia/GEM-X` on Hugging Face, governed by the NVIDIA Open Model License and the third-party notices published with the repository/model card.
- Related body model: `NVlabs/SOMA-X`, installed from the GEM-X-pinned submodule revision; Apache-2.0 source with its disclosed third-party identity-model terms.
- Storage: source, isolated environment, model assets, and generated checksum manifest remain under the gitignored `.kinetrace/engines/GEM-X` directory.
- Execution: local CUDA inference only; source videos are never uploaded by KineTrace. Setup downloads model weights from Hugging Face.
- Status: experimental selectable backend. Commercial/output-use conclusions remain the user's legal decision; this record is engineering due diligence, not legal advice.

This file records engineering due diligence and is not legal advice.
