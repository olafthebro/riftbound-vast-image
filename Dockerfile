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

# Hermetic python 3.12 venv with the exact validated stack
# (transformers pin is REQUIRED: unsloth's own 5.5.0 pin lacks gemma4_unified)
RUN uv venv /opt/riftbound-venv --python 3.12 \
 && VIRTUAL_ENV=/opt/riftbound-venv uv pip install \
      --python /opt/riftbound-venv \
      huggingface_hub \
      hf_transfer \
      unsloth \
      "transformers==5.16.0" \
 && /opt/riftbound-venv/bin/python -c \
      "import transformers, torch; print(f'transformers {transformers.__version__} torch {torch.__version__}'); import unsloth; print('unsloth OK')"

# Make the venv the default python for interactive/ssh use
ENV VIRTUAL_ENV=/opt/riftbound-venv
ENV PATH="/opt/riftbound-venv/bin:${PATH}"

# HF speed + xet-off defaults (mirrors setup_vast_a100.sh; token comes from
# .env.vast pushed per rent, never baked in)
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HUB_DISABLE_XET=1
