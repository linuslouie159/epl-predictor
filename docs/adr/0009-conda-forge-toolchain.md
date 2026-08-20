# Use Miniforge and conda-forge for the environment, not pip and venv

The machine has Python 3.12 and pip but no C or C++ compiler of any kind — no gcc, no clang, no MSVC,
no Visual Studio Build Tools. That rules out cmdstanpy, which requires a C++ toolchain, and degrades
PyMC's default PyTensor backend to a slow Python fallback. JAX has no official Windows binaries, so
NumPyro would put the project's most important dependency on its least-supported platform.

Miniforge with conda-forge is the path PyMC's own documentation recommends on Windows. It supplies
compilers and prebuilt scientific binaries, which removes an entire category of future build failure
rather than working around this one. `environment.yml` is the single source of truth for dependencies.

## Consequences

The existing plain Python 3.12 install is not used for this project. Contributors need conda rather
than pip alone, and package resolution is slower than pip's. In exchange, adding any future
scientific dependency — XGBoost for the deferred ML stage, arviz, Understat tooling — is expected to
just work.
