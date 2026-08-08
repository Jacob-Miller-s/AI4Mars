# Protocol Clarification Triage

## Scope And Guardrails

This document triages the completed intra-rater consistency evidence. It does not designate a correct pass, alter a human annotation, activate the proposed v2.1 protocol, or authorize pilot expansion, target export, or training.

The six-image clarification set is the minimum set that includes every structural discrepancy. `NLB_483340393EDR_F0470522NCAM00253M1` and `NLB_557555419EDR_F0652882NCAM00296M1` were visually inspected as count-stable controls and are not added because they introduce no distinct unresolved rule.

## Discrepancy Taxonomy

| Image | Observed evidence | Ontology or boundary | Evidence-supported category |
| --- | --- | --- | --- |
| `NLB_517255503EDR_F0541610NCAM07753M1` | Candidate 2 has three primary versus two repeat children; one primary child has no geometric overlap with repeat. Components 5 and 11 also change direct disposition. | Ontology | One formation split into multiple instances; missed or secondary-object handling |
| `NLB_483955685EDR_F0470598NCAM00320M1` | Component 3 is primary `rejected_bedrock` and repeat `accepted`; component 1 carries primary `uncertain` and `rejected_bedrock` records. | Ontology | Continuous Bedrock versus discrete rock; uncertain-region handling |
| `NLB_490004046EDR_F0482122NCAM00281M1` | Both passes subdivide component 1, but several children are unmatched; component 5 is primary accepted and repeat noise. | Ontology, then boundary | Weak inter-rock boundary; arbitrary subdivision of one coherent formation |
| `NLB_548252623EDR_F0631150NCAM00312M1` | Component 4 is primary noise and repeat accepted with best IoU `0.2045`. | Ontology | Missed small or secondary rock |
| `NLB_528261206EDR_F0580738NCAM00385M1` | Primary explicitly merges components 3 and 4 with note `3 and 4 same rock`; repeat rejects 3 and accepts 4 without a merge. | Ontology | Multiple formations merged in one pass; semantic candidate fragmentation influencing object accounting |
| `NLB_463551084EDR_F0411534NCAM00385M1` | Component 2 contributes 13 primary versus 5 repeat accepted instances, all within a long layered visible formation. | Ontology, then boundary | Arbitrary subdivision of one coherent formation; weak or hidden inter-rock boundary |

## Dedicated 15-Versus-6 Audit

`NLB_463551084EDR_F0411534NCAM00385M1` contains 15 primary and 6 repeat accepted instances. Component 2 accounts for 13 primary and 5 repeat accepted instances. The RGB overlays show a long layered surface where multiple primary polygons lie along the same visible formation, while repeat uses fewer, larger polygons. This is not evidence that either pass is correct; it identifies the missing rule: a layered band or tonal/texture variation is not sufficient by itself to create a new object.

| Primary instance | Source component(s) | Best repeat match | IoU | Triage category |
| --- | --- | --- | ---: | --- |
| `rock-001` | `[1]` | none; repeat component 1 is rejected noise | 0.0000 | Primary-only object; rock/noise disposition ambiguity |
| `rock-003` | `[4]` | `rock-008` | 0.6894 | Boundary-placement disagreement without count change |
| `rock-004` | `[2]` | `rock-002` | 0.3916 | Subdivision or weak-boundary disagreement |
| `rock-005` | `[2]` | `rock-003` | 0.7257 | Boundary-placement disagreement |
| `rock-006` | `[2]` | `rock-004` | 0.5437 | Boundary-placement disagreement |
| `rock-007` | `[2]` | none | 0.0000 | Primary-only subdivision |
| `rock-008` | `[2]` | `rock-005` | 0.7000 | Boundary-placement disagreement |
| `rock-009` | `[2]` | none | 0.0000 | Primary-only subdivision |
| `rock-010` | `[2]` | none; best overlap is 0.0335 | N/A | Weak-boundary/subdivision disagreement |
| `rock-011` | `[2]` | `rock-006` | 0.1396 | Weak-boundary/subdivision disagreement |
| `rock-012` | `[2]` | none; best overlap is 0.0451 | N/A | Weak-boundary/subdivision disagreement |
| `rock-013` | `[2]` | none; best overlap is 0.1261 | N/A | Weak-boundary/subdivision disagreement |
| `rock-014` | `[2]` | none | 0.0000 | Primary-only subdivision |
| `rock-015` | `[2]` | none | 0.0000 | Primary-only subdivision |
| `rock-016` | `[2]` | none | 0.0000 | Primary-only subdivision |

All six repeat accepted objects have a primary geometric match; the evidence is asymmetric because the primary pass subdivides component 2 more finely. This is why the clarification review must answer the visible-separator question before considering polygon precision.

## Isolated Human Clarification Review

The candidate manifest [clarification_review_v2.1-candidates.csv](../../../research/rock_instance/clarification_review_v2.1-candidates.csv) contains six images and the exact human question for each. Prior primary and repeat annotations must be hidden. For each image, review RGB and terrain context, make full per-component terminal decisions, and record a split or merge only when the proposed visible-separator rules support it.

The existing interactive reviewer already supplies RGB, NAV context, candidate references, reviewer polygons, direct terminal dispositions, and merge records. No UI change is required. The new initializer creates a separate empty state and preserves the source v2.0 protocol, proposed-v2.1 hash, primary/repeat hashes, agreement-report hash, and selection-manifest hash.

After human review, compare the clarification state with both prior passes. Recheck object identity first, then accepted count and split/merge structure, then polygon overlap for matched objects. The clarification succeeds when targeted cases receive a defensible, repeatable visible interpretation without introducing a new contradiction; no perfect-IoU or universal threshold is required.