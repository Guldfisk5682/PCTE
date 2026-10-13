# Official VLM DA adapters

Only the added MTDA adapter files are stored here. Download the original code
from its authors with `python scripts/setup_vlm_competitors.py`. The installer
pins AD-CLIP to `06a30ce32ab7a0dcc29a5ac72e66c4e868f70c89` and DAPrompt to
`f24b0c16c06b585104ba97615f89cd5df4def872`, verifies native trainer files, and
refuses to overwrite a different existing adapter.

The archived files are byte-identical to those used for the reported runs.
Their original sibling wrappers assume this project is named `spl`; when the
checkout is named `PCTE`, invoke its `train_vlm_competitors.py` or matrix runner
directly with `--adclip-repo ../AD-CLIP --daprompt-repo ../DAPrompt`.

See [the protocol](../docs/vlm_competitors_protocol.md) for provenance, native
objectives, implementation limitations, and controlled comparison budgets.
