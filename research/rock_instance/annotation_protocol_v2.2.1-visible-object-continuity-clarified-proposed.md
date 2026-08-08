# Rock Instance Annotation Protocol v2.2.1 - Visible Object Continuity Clarified (Proposed)

## Status

This proposed revision supplements proposed v2.2. It is not active or frozen. It does not change historic annotations, accepted identity, disposition, split/merge decisions, review scope, training targets, or model training.

Protocol identifier: `v2.2.1-visible-object-continuity-clarified-proposed`

## Annotation Target

For an already accepted rock, the polygon represents the defensible visible image-plane extent of that one physical rock object. Include every pixel whose visible surface is defensibly part of that object, regardless of semantic-proposal coverage.

## Whole Object vs. Face Rule

When a rock has already been accepted as one physical object, annotate the full defensible visible image-plane extent of that physical object.

Do not create or retain a smaller instance merely because one surface face has stronger contrast, clearer texture, a stronger local contour, different illumination, or a semantic proposal covering only that face. A visible face is not a separate object unless independent RGB evidence supports a separate physical-rock identity.

## Continuity Rule

Treat exposed rocky regions as belonging to the same accepted object when RGB evidence supports a coherent continuous physical form and outer contour. Internal shading, texture, stratification, planar-face, or surface-normal changes do not by themselves terminate the object mask.

## Object Termination Rule

Terminate the visible mask at a defensible external object contour, surrounding sand or soil, continuous Bedrock not belonging to the object, another independently identified rock, rover hardware or another occluder, or an edge where object membership becomes genuinely indeterminate.

## Shadow Rule

Cast shadow is not part of the object mask. Shadow does not by itself establish a new object boundary. Where visible geometry supports continuity around or across a shadowed region, preserve the single accepted identity and annotate only pixels visibly attributable to the rock; do not include unobservable shadow pixels by interpolation.

## Occlusion Rule

Where rover hardware or terrain occludes the rock, stop the mask at the visible occlusion boundary and do not infer hidden pixels. Do not reduce the object to one small exposed face merely because another visible portion is partly occluded.

## Conservative Ambiguity Rule

If adjacent rocky material cannot be defensibly assigned to the accepted object, do not invent a connection. Use the smallest extent supported by coherent external-boundary evidence, while distinguishing that decision from selecting only a visually salient face.

## Final Clarification Prompt

Trace the full defensible visible extent of this already-accepted physical rock. Include all visibly attributable faces of the same coherent object. Do not trace only a high-contrast face or proposal fragment. Stop at surrounding terrain, continuous Bedrock, another rock, rover hardware/occlusion, or genuinely indeterminate material. Exclude cast shadow and do not infer hidden geometry.