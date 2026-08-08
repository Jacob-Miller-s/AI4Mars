# Rock Instance Annotation Protocol v2.1 - Object Identity Clarified (Proposed)

## Status

This is a proposed clarification to `v2.0-calibration-resolved`. It is not active, frozen, or authorized for the remaining pilot images, target export, or training. It was prompted by the completed eight-image intra-rater consistency analysis, especially the 15-versus-6 instance-count divergence in `NLB_463551084EDR_F0411534NCAM00385M1`.

Existing primary and repeat annotations remain historical evidence. This proposal does not reinterpret, overwrite, or select either pass as correct.

## Proposed Object-Identity Rules

### One Visible Rock Instance

Annotate one instance for one visibly coherent, discrete rock surface. Determine identity from RGB-visible object boundaries, not from semantic candidate-component connectivity, bounding boxes, or polygon convenience.

Treat adjacent lobes as one instance when the visible surface remains coherent and there is no observed terrain gap, overlap/occlusion edge, or clear separative contour between them. A tonal change, cast shadow, texture change, layered band, shallow crease, or the end of a semantic candidate component does not by itself establish two objects.

### When To Split

Split only when the RGB view shows a defensible visible separator between discrete exposed objects: a gap of surrounding terrain, an overlap/occlusion relationship, or a stable boundary that separates independently visible outer contours. Do not infer attachment, separation, or hidden geology beneath burial or shadow.

When a source component yields more than one accepted object, record an explicit split resolution linked to the child annotations. A direct component decision cannot contain several accepted children without a split record.

### When To Merge

Merge semantic components only when the RGB view supports one coherent visible object across their component boundary. Record the contributor set and visible rationale. Do not merge merely because components touch in the semantic mask.

### Weak Boundaries And Uncertainty

If a suspected separator is only a weak tonal/shadow/texture feature and does not support a defensible visible object identity, do not draw an arbitrary split. Use one coherent instance when its outer boundary is visible; otherwise use `uncertain` with a short rationale rather than forcing split or merge.

Each source component must receive one terminal direct disposition unless an explicit split or merge resolution accounts for it. Do not retain incompatible terminal dispositions for the same component outside a resolution record.

### Bedrock Versus Discrete Rock

Accept a discrete rock only when it has a visibly bounded exposed object surface distinct from the surrounding continuous terrain. A continuous slab, outcrop, or terrain-connected surface without a defensible outer object boundary is `rejected_bedrock`. If the available RGB evidence cannot support either conclusion, use `uncertain`; do not infer a subsurface boundary.

### Partially Buried, Occluded, And Small Features

For a partially buried object, identity requires a distinct exposed outer contour and visible separation from neighboring surfaces. Do not create a second object from a protrusion that remains visibly part of the same coherent surface. Record `truncated` or `occluded` only for an accepted object whose visible extent reaches the image edge or is blocked by another visible object; these flags do not create identity evidence.

## Proposed Boundary-Drawing Rules

Draw only the visible exposed extent of an accepted object. Stop at the visible boundary against sand, Bedrock, shadowed occlusion, or another accepted object. Do not extend a polygon through buried or occluded regions.

For a short hidden edge between otherwise visibly separate touching rocks, interpolate only the shortest local connection needed to close the visible object boundary; do not use that interpolation as evidence that the objects are separate. For touching rocks with no defensible separator, draw one outer boundary for the coherent object.

Exclude surrounding sand and Bedrock from the polygon. Include an attached-looking protrusion only when it is visibly part of the same coherent surface under the identity rules above. Polygon precision is secondary to the object-identity decision and must not change the object count by itself.

## Calibration Evidence And Expected Effect

| Evidence | Proposed clarification | Expected effect | Prior annotation handling |
| --- | --- | --- | --- |
| `NLB_463551084EDR_F0411534NCAM00385M1`: component 2 produced 13 primary versus 5 repeat accepted instances | Require visible separators and explicit split records for multiple children | Prevent arbitrary subdivision of a coherent layered formation | Preserve both passes; test only in isolated clarification review |
| `NLB_528261206EDR_F0580738NCAM00385M1`: primary merge `[3, 4]`, repeat direct component 4 acceptance | Require visible merge rationale and contributor provenance | Makes merge-versus-separate interpretations auditable | Preserve both passes; no automatic merge choice |
| `NLB_483955685EDR_F0470598NCAM00320M1`: component 3 changed Bedrock to accepted and component 1 had incompatible terminal history | Define RGB-visible discrete-object evidence and one-terminal-direct-disposition rule | Separates ontology decisions from component coverage mechanics | Preserve both passes; seek human clarification |
| `NLB_490004046EDR_F0482122NCAM00281M1` and `NLB_517255503EDR_F0541610NCAM07753M1`: repeated child-count differences within one candidate | Require explicit split records and visible separator evidence | Reduces component-driven child-count variation | Preserve both passes; no count target is imposed |

## Activation Boundary

Human approval is required before this proposal is used for any clarification review. After isolated clarification review, compare object identity, count, split/merge structure, and visible-boundary agreement against both prior passes. Success means the targeted cases can be interpreted under the clarified rules without a new contradiction; it does not require perfect mask IoU or a universal numeric threshold.