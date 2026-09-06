# Riftbound Vast.ai bootable env image
#
# Bakes the FULL training environment so a rented Vast instance boots ready:
#   pytorch/pytorch:2.6.0 (same base the validated runs used)
#   + uv + python 3.12 venv with unsloth + transformers==5.16.0 + hf_transfer
#
# Deliberately does NOT contain: model weights (24GB — kept on the Vast
# volume), datasets, or any project scripts (rsync'd per rent). Keeping the
# image lean means fresh-host pulls stay as small as possible and nothing
# proprietary ships in a public package.
#
# Runtime layout convention (see docs/VAST-VOLUME-RUNBOOK.md):
#   /opt/riftbound-venv        <- this image's venv
#   ~/riftbound-train          <- symlink -> volume mount (/workspace)
#   ~/riftbound-train/.venv    <- symlink -> /opt/riftbound-venv
# setup_vast_a100.sh detects the image venv and skips the pip build; it only
# downloads the model into the volume once, then writes .setup_ok.

# NOTE: plain pytorch/pytorch:2.6.0 does NOT exist on Docker Hub (404) —
# only suffixed tags. Vast's morning host served it from cache; fresh hosts
# hang in "loading" forever trying to pull it. Always use the explicit tag.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# uv (fast python env manager). Installed via pip, NOT the curl|sh script —
# the pipe silently masks curl failures and left uv missing (exit 127).
RUN python -m pip install --no-cache-dir uv

# git is required by uv to install unsloth/unsloth-zoo from git URLs
# (the pytorch base image does not ship git)
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Hermetic python 3.12 venv with the exact validated stack.
# 🔴 2026-09-06 lessons baked in:
#   - unsloth 2025.9.5 (PyPI) is BROKEN with fresh torch (auto_docstring exec
#     NameError + torch-version guards) -> install CURRENT unsloth from git,
#     paired with unsloth-zoo from git (PyPI zoo is stale, missing device_type).
#   - transformers pin is REQUIRED: unsloth's dep pin (5.5.0) lacks
#     gemma4_unified -> re-pin 5.16.0 AFTER the git install.
#   - torch pinned to 2.11.0: 2.14.0 (released later the same day) breaks
#     unsloth. 2.11.0+cu130 was the version on the WORKING morning run.
#   - unsloth cannot be IMPORTED at build time (no NVIDIA GPU on GHA runners)
#     -> verify presence with find_spec; real import check runs on the host.
RUN uv venv /opt/riftbound-venv --python 3.12 \
 && VIRTUAL_ENV=/opt/riftbound-venv uv pip install \
      --python /opt/riftbound-venv \
      huggingface_hub \
      hf_transfer \
      "unsloth @ git+https://github.com/unslothai/unsloth.git" \
      "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git" \
 && VIRTUAL_ENV=/opt/riftbound-venv uv pip install \
      --python /opt/riftbound-venv \
      "torch==2.11.0" \
      "transformers==5.16.0" \
 && /opt/riftbound-venv/bin/python -c \
      "import transformers, torch; print(f'transformers {transformers.__version__} torch {torch.__version__}'); import importlib.util; assert importlib.util.find_spec('unsloth'), 'unsloth missing'; print('unsloth installed OK')"

# Make the venv the default python for interactive/ssh use
ENV VIRTUAL_ENV=/opt/riftbound-venv
ENV PATH="/opt/riftbound-venv/bin:${PATH}"

# HF speed + xet-off defaults (mirrors setup_vast_a100.sh)
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HUB_DISABLE_XET=1

# ==== Bake the 24GB gemma4 base model (2026-09-06) ====
# Fresh hosts then boot with env + model ready; only the ~250MB adapter gets
# uploaded per run. Token comes via a buildkit SECRET so it never lands in an
# image layer (a public image + build-arg would leak it).
# NOTE: hosts must have driver >= 570 (the venv's torch is the +cu130 build).
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) \
    /opt/riftbound-venv/bin/python -c \
      "from huggingface_hub import snapshot_download; \
       snapshot_download('unsloth/gemma-4-12b-it', local_dir='/opt/gemma-4-12b-it', max_workers=8); \
       print('model baked at /opt/gemma-4-12b-it')"
# Runtime layout (no volume needed): workdir lives on the container disk;
# onstart symlinks wire the baked env + model into ~/riftbound-train:
#   ln -sfn /opt/riftbound-venv   /root/riftbound-train/.venv
#   ln -sfn /opt/gemma-4-12b-it   /root/riftbound-train/models/gemma-4-12b-it
