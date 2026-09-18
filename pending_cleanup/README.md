# Pending Cleanup

This folder preserves files that were moved out of the active repository root
during the FLARE dashboard refactor.

## Contents

- `legacy_rest_federation/`: old REST hub/worker prototype, old root Docker
  files, old shell runners, and old prototype tests.
- `legacy_docs/`: old root-level summaries, slide notes, and run notes.
- `docs/project_docs/`: older project documentation and architecture notes.
- `generated_artifacts/`: previously committed model outputs, metrics, images,
  worker reports, and old `training_docker_v1/checkpoints`.
- `reference_federated/`: duplicate top-level federated/FLARE reference code
  that is not part of the active `training_docker_v1` dashboard stack.
- `misc/`: miscellaneous root artifacts such as `.DS_Store`.

## Active Code

The current runnable FLARE dashboard lives in:

```text
../training_docker_v1
```

The core model/data utilities remain in:

```text
../training
```

Files in this folder can be reviewed later and deleted once the team confirms
they are no longer needed.
