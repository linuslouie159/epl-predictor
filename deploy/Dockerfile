# The live loop's environment, and nothing else (issue #19).
#
# This image is the *environment*, not the project. The repository is bind-mounted over /repo at
# run time (deploy/docker-compose.yml), so `git pull` on the Pi updates the code with no rebuild
# and the sealed rounds the loop writes land in a real checkout the owner can inspect and push by
# hand. What the image carries is the conda environment ADR 0009 chose, which is the part that is
# slow to build and has no business being rebuilt weekly.
#
# WHY A CONTAINER AT ALL, on a Pi that already runs another project on bare metal. The Pi's other
# tenant is a paper-trading loop installed as a venv of pip wheels, deliberately chosen so that
# nothing on that box needs a toolchain. This project cannot be installed that way: ADR 0009 needs
# conda-forge for the prebuilt scientific stack, and environment.yml pins a BLAS provider because
# the wrong one aborts the interpreter on every LAPACK call. Two package managers arguing over one
# machine's system libraries is precisely what an image prevents, and it is the only reason this
# one exists.
#
# BUILD (from the repository root, so the build context is the project):
#   docker build -f deploy/Dockerfile -t epl-predictor:live .
#
# Building natively on a Pi 5 takes a while — the conda solve is the slow half. Cross-building on a
# desktop and pushing the result is usually the better trade:
#   docker buildx build --platform linux/arm64 -f deploy/Dockerfile -t epl-predictor:live --load .

# Miniforge rather than any other base, because it is conda-forge's own image and ADR 0009 chose
# conda-forge for the prebuilt scientific stack. It is multi-arch, so this builds on aarch64.
#
# `:latest` is a floating tag, and this repository has been bitten twice by a free version choice —
# arviz 1.x removing what pymc imports, and conda picking an MKL whose threading layer aborts the
# interpreter on every LAPACK call (environment.yml says so at length). The exposure here is
# smaller than it looks: everything that actually matters is solved from environment.yml, which is
# the pinned surface, and the base contributes little beyond Debian and a conda. It is not zero.
# If a rebuild ever produces an image that behaves differently from the last one, pin this line to
# a digest before looking anywhere else.
FROM condaforge/miniforge3:latest

# UK local time inside the container, so that anything reading local time agrees with the loop's
# own clock rather than quietly disagreeing with it.
#
# Nothing in the code depends on it: `epl.ledger.live.uk_now` converts from UTC explicitly, so the
# sealing window is judged identically whatever TZ says — and the `===== RUN =====` stamps in
# deploy/logs/ are written by the *host* shell, not in here. So this is belt to the code's braces
# and is worth knowing about for one reason: it means changing TZ cannot fix a scheduling problem,
# though it will look like the sort of thing that would.
ENV TZ=Europe/London

# git is what makes a Sealed Prediction evidence (ADR 0005) — the loop commits and pushes through
# it — and openssh-client is what lets that push reach a remote over a deploy key. Neither is in
# the base image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y git openssh-client tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo

# The whole of environment.yml goes in, and the live loop needs about half of it. Measured: with
# every Predictor registered and `epl.live` imported, matplotlib, PyMC and arviz are all absent from
# sys.modules — they belong to `epl.simulate`, which the loop never calls. So a slimmer image is
# available and was not taken, deliberately: parity means this Pi can also run a Season Projection
# when there is one to run, and a second environment definition that drifts from environment.yml is
# a worse problem than a larger image. If build time on the Pi ever becomes the binding constraint,
# dropping pymc/nutpie/arviz/gxx/matplotlib is the lever, and giving up `epl.simulate` is its price.
#
# The conda environment is built in one layer and the project is installed into it in another, so
# that editing a line of Python does not re-solve two hundred packages. environment.yml stays the
# single source of truth (ADR 0009); what is stripped here is only its trailing editable install,
# which is the project itself and belongs in the later layer.
#
# The grep is a guard rather than decoration. If that pip section ever moves or is renamed, the
# sed below would silently stop stripping anything, `conda env create` would install the project
# from a build context that is not there yet, and the failure would surface as a confusing solve
# error rather than as the thing that actually changed.
COPY environment.yml ./
RUN grep -q '^      - --editable \.$' environment.yml \
    && sed '/^  - pip:/,$d' environment.yml > /tmp/environment-no-project.yml \
    && conda env create -f /tmp/environment-no-project.yml \
    && conda clean --all --yes

# Every later command runs inside the environment. `conda run` would work too and swallows the
# exit code in ways a schedule must not tolerate, so the environment's bin directory goes on PATH
# instead and `python` is simply the right one.
ENV PATH=/opt/conda/envs/epl-predictor/bin:$PATH

# The editable install. It records a path — /repo/src — rather than copying anything, which is
# what makes the bind mount work: at run time that path is the host's checkout, so the code the
# loop executes is the code the Pi has pulled, not the code this image was built from.
# README.md is deliberately not copied: pyproject.toml declares no `readme`, so bringing it in
# would only invalidate this layer every time the documentation changed.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps --editable .

# The layout module resolves everything from here (epl.paths), and the bind mount lands here.
ENV EPL_PROJECT_ROOT=/repo

# No CMD that writes. `upcoming` is the one command in the loop that fetches, reports and writes
# nothing at all, so a container started with no arguments answers the project's open question
# instead of sealing something nobody asked it to.
ENTRYPOINT ["python", "-m", "epl.live"]
CMD ["upcoming"]
