# Rock Instance Annotation Protocol v2.2 - Visible Extent Clarified (Proposed)

## Status

This proposed revision supplements the proposed v2.1 object-identity clarification. It is not active or frozen. It does not change any historic annotation, authorize further pilot review, create training targets, or permit model training.

## Annotation Target

For an already accepted rock, the polygon represents **the visible image-plane extent of that physical rock object**. Include every pixel whose visible surface is defensibly part of the accepted object, regardless of whether it lies inside the semantic proposal component. The polygon is neither the semantic component nor only the most salient top or front face.

Do not infer pixels hidden beneath sand, soil, another object, rover hardware, or terrain. Cast shadow is not rock surface and is excluded unless visible rock material can independently be traced through it.

## Proposed Boundary Rules

1. **Whole visible object.** Trace the entire defensible visible extent of the accepted rock, including visible side, top, and front surfaces that belong to that one object.
2. **Semantic proposal is non-binding.** A connected component and its box are proposal/provenance evidence only. They must not clip or expand the polygon.
3. **Shadow exclusion.** Do not include cast shadow as rock pixels. Use material, contour, and contact evidence to trace the rock edge rather than the shadow edge.
4. **Partial burial.** Stop at the visible contact with sand or soil. Do not extrapolate beneath burial, even where the buried continuation is physically plausible.
5. **Occlusion.** Stop at rover hardware, terrain, or another rock that blocks the surface. Do not hallucinate hidden pixels behind an occluder.
6. **Touching objects.** Once object identity has established separate rocks, each polygon stops at the visible shared or occluded interface and contains only the visible surface belonging to that object.
7. **Bedrock exclusion.** Do not extend an accepted-rock polygon into continuous Bedrock/outcrop merely because illumination, texture, or stratification is similar.
8. **Weak local edge.** When a short edge is weak but the accepted object is otherwise visible, use the most conservative local contour supported by adjacent visible boundary evidence. Do not create a large hidden continuation.

## Boundary Escalation

If the fixed target reference cannot be located as one defensible accepted object without reopening object identity, record an identity escalation instead of changing `accepted` to another status. This is an audit signal, not a relabeling mechanism.

## Recheck Boundary

The isolated redraw must be compared with primary, repeat, and v2.1 masks. Evaluate visible-surface reasoning, area, bbox, overlap relationships, and qualitative compliance. No universal IoU threshold decides success; remaining differences must be ordinary contour precision rather than different interpretations of which visible pixels belong to the object.