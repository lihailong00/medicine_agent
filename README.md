# Medicine Agent

Offline-first CLI/library prototype for a bioinformatics scientific research agent.
Live literature retrieval is available only through an explicit flag and is
hard-restricted to PubMed/NCBI E-utilities, arXiv API, and Semantic Scholar API.

## Run

```bash
python -m medicine_agent.cli run \
  --question "Which ligand-receptor interactions are most relevant?" \
  --data-dir data \
  --output-dir generated/medicine_agent \
  --offline
```

The default/offline pass is deterministic and requires no API keys or new
dependencies. Original `data/` files are read-only inputs; derived
reports/manifests/tables are written under the chosen output directory.

For real network literature retrieval, opt in explicitly:

```bash
PYTHONPATH=src python -m medicine_agent.cli run \
  --question "Which ligand-receptor interactions are most relevant in tumor immune communication?" \
  --data-dir data \
  --output-dir generated/medicine_agent \
  --live-api
```

`--live-api` performs real HTTPS calls only to:

- `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/` for PubMed/NCBI ESearch and EFetch
- `https://export.arxiv.org/api/query` for arXiv
- `https://api.semanticscholar.org/graph/v1/paper/search` for Semantic Scholar

`--offline` always forces fixture mode and prevents live calls, even if
`--live-api` is also present.

## Safety and Evidence Policy

- All side-effectful actions pass through `SafetyGate`.
- Network calls are allowed only for PubMed/NCBI, arXiv, and Semantic Scholar API hosts; all other hosts are blocked.
- Dependency install, API key use, long jobs, script execution, and overwrites require confirmation in non-interactive mode.
- Biomedical output is research-only and not clinical decision support.
- Synthesis claims must carry a `ClaimStatus` and evidence references unless explicitly labeled as a hypothesis or out-of-scope clinical refusal.
