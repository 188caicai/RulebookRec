# RulebookRec

RulebookRec combines retrieval evidence with semantic USER and ITEM rules through a Ranker, Reflector, and Governor.

## Method

The Ranker reads the latest five visible interactions, candidate metadata, and the frozen rulebook. It annotates each candidate with a USER group (`primary`, `secondary`, `edge`, or `none`) and supported ITEM-rule matches. Deterministic code orders the fixed candidate set by:

1. USER group: primary, secondary, edge, then none;
2. whether at least one compatible ITEM rule is matched;
3. number of supporting retrieval tools;
4. weighted reciprocal-rank fusion rank.

Candidates with any ITEM match share the same second-tier priority. Helpful/Harmful counts inform semantic assessment in the Ranker.

After the ranking is finalized, the observed interaction becomes available. The Reflector assesses every cited USER rule and independently considers a USER proposal and an ITEM proposal. ITEM analysis uses the highest ranked predecessor in the target's USER group; it is skipped when that comparison item is unavailable.

The Governor reviews the original evidence in a separate call and chooses `ADD`, `MERGE`, or `REJECT`. ADD assigns a new ID with zero counts. MERGE appends supporting evidence to an equivalent retained rule while preserving its ID, content, interest boundary, and accumulated counts. Code applies the current event's audit increments separately.

Auditing uses the frozen rule versions and the Governor's support judgments. A USER rule receives Helpful for target support and Harmful for target mismatch. An ITEM rule receives Helpful when it supports the target and the comparison judgment is definite; it receives Harmful when it supports the comparison item but not the target. Uncertain judgments leave counts unchanged. Each rule is updated at most once per event, and new rules become eligible in the next round.

The next round uses the updated rulebook and history including the observed interaction. Final testing uses retrieval, Ranker annotation, and deterministic reranking with a frozen rulebook.

## Core interfaces

`WeightedRRFFuser` combines retrieval lists with smoothing constant 100 and returns 20 unseen candidates. `ExternalRetrieverAdapter` accepts a callback returning the retrieval views. `build_external_module_adapters` connects the three prompt templates to a JSON-call function; use the same LLM backbone with temperature 0 for these calls.

```python
from rulebookrec import (
    Governor, InMemoryGovernorStore, build_external_module_adapters,
)

modules = build_external_module_adapters(call_json)
governor = Governor(
    store=InMemoryGovernorStore(initial_snapshot),
    adapter=modules.governor,
)
```

The JSON-call function has signature `call_json(prompt_text, payload)` and returns the structured object requested by that prompt. `EvolutionPipeline.run_round` receives visible history, a callback revealing the observed interaction, and a unique audit event ID. Its caller appends that interaction to the history before the following round. `FinalTestPipeline.run` receives the observed history and frozen rulebook.

## Repository contents

```text
prompts/             Ranker, Reflector, and Governor prompts
src/rulebookrec/     Schemas, retrieval fusion, ranking, reflection, and governance
requirements.txt     Dependencies
README.md
```

## Installation

The implementation uses Python 3.9.12 and PyTorch 1.12.1.

```powershell
conda create -n rulebookrec python=3.9.12 -y
conda activate rulebookrec
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD/src"
```
