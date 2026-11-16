# N018 checkpoint

`best.pt` is the externally trained N018 model selected in the research record.
The team confirmed that the model may be shared for the competition. The file
does not include the challenge dataset; it contains model tensors, architecture
and training configuration, the selected step, and external validation metrics.

SHA-256: `1b42d0f7eab232f6ba4ea2131d3f0a892f41f750504789dfcd5ce5edb5158a81`

Use it with the frozen decoder at `configs/neural/n018_decoder.json` and the
inference instructions in the repository README. PyTorch checkpoints use
serialization capable of executing code when loaded; use only the reviewed file
from a trusted source. The challenge data has separate access and usage terms.
