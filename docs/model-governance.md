# Model governance

Generated animation may be used in a commercial downstream project, so model selection is treated as a commercial-compatibility requirement even though KineTrace itself is personal software.

The first backend uses official Google MediaPipe model bundles. Before adding any alternative model, record:

- source repository and exact version;
- code license;
- model-weight terms;
- training-dataset restrictions disclosed by the provider;
- whether commercial use and generated-output use are addressed;
- model checksum.

An open-source implementation license does not automatically determine the terms of its checkpoints or training data. Alternative MMPose checkpoints will therefore be audited individually before they become a selectable backend.

This file records engineering due diligence and is not legal advice.
