# Medicine Agent

Offline-first CLI/library prototype for a bioinformatics scientific research agent.

## Run

```bash
python -m medicine_agent.cli run \
  --question "Which ligand-receptor interactions are most relevant?" \
  --data-dir data \
  --output-dir generated/medicine_agent \
  --offline
```

The first pass is deterministic and requires no API keys or new dependencies. Original `data/` files are read-only inputs; derived reports/manifests/tables are written under the chosen output directory.

## Safety and Evidence Policy

- All side-effectful actions pass through `SafetyGate`.
- Network/live API, dependency install, API key use, long jobs, script execution, and overwrites require confirmation in non-interactive mode.
- Biomedical output is research-only and not clinical decision support.
- Synthesis claims must carry a `ClaimStatus` and evidence references unless explicitly labeled as a hypothesis or out-of-scope clinical refusal.
