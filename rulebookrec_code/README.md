# RulebookRec

This repository implements the core RulebookRec method: semantic rule
annotation,  four-tier reranking, trace-guided reflection, and
governed two-level rule evolution.

## Method overview

```text
Tail5 history + fixed candidate slate + stored USER -> ITEM rule tree
                              |
                              v
Ranker: contextualize rules and annotate candidate-rule matches
                              |
                              v
Deterministic four-tier reranker
  1. ordered USER-rule match vector
  2. eligible ITEM-rule match vector
  3. retrieval-tool consensus
  4. raw fusion rank
                              |
                              v
Reveal the training target and construct a deterministic decision trace
                              |
                              v
Reflector: diagnose the failed tier and propose the smallest reusable repair
                              |
                              v
Governor: ADD / UPDATE / MERGE / REJECT, optionally return revision feedback,
          audit Helpful/Harmful, and commit the next rulebook snapshot
```

The Ranker does not output a permutation. Given its semantic annotations, final
candidate ordering is produced by deterministic code.

## Two-level rulebook

RulebookRec stores two complementary types of preference memory:

- **USER rules** are recurring upper-level interests shared across recent
  interactions. They determine the ordered macro-preference routing vector.
- **ITEM rules** are transferable fine-grained choice boundaries attached to
  one or more stored USER parents. They refine alternatives only after their
  parent USER conditions are matched.

Every rule stores `Helpful` and `Harmful` counts. The counts indicate historical
reliability and help order rules with comparable current relevance; they do not
form an additional reranking tier.

## Ranker

The Ranker receives:

- the ordered Tail5 interaction history;
- the complete stored USER-to-ITEM hierarchy, including rule state and credit;
- semantic content for every candidate, identified by opaque keys.

It first infers the current preference context, orders active USER rules, and
then annotates which USER rules each candidate realizes. A candidate may match
multiple USER branches. ITEM children are considered only when all of their
stored USER parents are matched. Domain metadata is treated as semantic evidence
rather than a closed ontology.

## 4-tiers reranking

For candidate `c`, the reranker constructs

```text
K(c) = (-U(c), -I(c), -T(c), R_raw(c))
```

where `U(c)` is the binary USER-match vector in the Ranker's contextual order,
`I(c)` is the eligible ITEM-match vector, `T(c)` is retrieval-tool support, and
`R_raw(c)` is raw fusion rank. Candidates are sorted lexicographically by this
key. With an empty rulebook, ordering deterministically reduces to tool consensus
followed by raw fusion rank.

## Reflector

After reranking, the training target is revealed. The
Reflector receives Tail5, target semantics, the direct routing trace, relevant
stored rules, and a valid comparator when available.

Reflection follows the observed failure type:

- a correct Top-1 retains evidence without creating a rule;
- a USER-routing failure considers one reusable USER repair;
- a within-routing-signature failure considers one ITEM repair under stored
  USER parents;
- absence of a valid ITEM comparator produces no ITEM proposal.

A USER proposal must express a recurring upper-level pattern shared by Tail5 and
the target. An ITEM proposal must express a transferable target-versus-comparator
boundary within the same stored USER interest and be independently supported by
recent history. Existing memory is repaired or consolidated before a new rule is
added. A Ranker matching error may leave the rulebook unchanged.

## Governor

Governor is the only component that writes rule state. It reviews each proposal
against the frozen tree and chooses:

- `ADD` for a genuinely missing reusable preference;
- `UPDATE` for an inaccurate or non-discriminative stored boundary;
- `MERGE` for duplicate or overlapping stored memory;
- `REJECT` for unsupported, item-specific, overly broad, or non-transferable
  memory.

A rejected proposal may be returned to Reflector with concrete feedback for one
revision over the same frozen trace. Only the final accepted review is committed.
USER merges preserve and reattach ITEM children; updates preserve rule identity
and accumulated credit.

Helpful/Harmful auditing uses stored rules involved in the observed decision and
the revealed outcome. Newly admitted rules are first eligible in a later round.
Final testing freezes the rulebook and contains no Reflector or Governor write
path.

## Repository contents

```text
prompts/             Ranker, Reflector, and Governor prompts
src/rulebookrec/     Schemas, rule annotation, reranking, reflection, governance
requirements.txt     Three-tool training and runtime dependencies
README.md
```

## Installation

The main dependencies for CSR (LightGCN), USR (SASRec), SEM retrieval,
and RulebookRec execution are listed in `requirements.txt`. The implementation
is developed using Python 3.9.12 and PyTorch 1.12.1.

```powershell
conda create -n rulebookrec python=3.9.12 -y
conda activate rulebookrec
python -m pip install -r requirements.txt
```
