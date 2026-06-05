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

To request approved full-text evidence after live metadata search, add
`--full-text`:

```bash
PYTHONPATH=src python -m medicine_agent.cli run \
  --question "Intercellular communication analysis of single-cell transcriptomics data" \
  --data-dir data \
  --output-dir generated/medicine_agent \
  --live-api \
  --full-text
```

`--full-text` requires `--live-api` and cannot be combined with `--offline`.
It still uses only approved source routes:

- NCBI E-utilities PubMed→PMC ELink and PMC EFetch XML when a PMCID/PMC link is available
- Constructed `https://arxiv.org/pdf/<arxiv-id>` PDF artifact URLs for arXiv records
- `https://api.semanticscholar.org/graph/v1/snippet/search` snippets for Semantic Scholar records

The agent writes `artifacts/full_text_results.json` plus per-paper text/PDF
artifacts where retrieval succeeds. arXiv PDFs are saved as audit artifacts but
are not parsed as dependency-free full text.

`--offline` always forces fixture mode and prevents live calls, even if
`--live-api` is also present.

## Safety and Evidence Policy

- All side-effectful actions pass through `SafetyGate`.
- Network calls are allowed only for PubMed/NCBI, arXiv, and Semantic Scholar API hosts; all other hosts are blocked.
- Full-text retrieval never follows publisher or arbitrary `openAccessPdf` URLs.
- Dependency install, API key use, long jobs, script execution, and overwrites require confirmation in non-interactive mode.
- Biomedical output is research-only and not clinical decision support.
- Synthesis claims must carry a `ClaimStatus` and evidence references unless explicitly labeled as a hypothesis or out-of-scope clinical refusal.
