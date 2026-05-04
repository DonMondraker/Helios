# CAD_Image_automation

Step-by-step: what happens in the background

1. Image upload

The user uploads a screenshot.
The program loads:
* the original image
* and creates a normalized working image
  The original image is only for reference.
  The normalized image is what the system actually uses for processing.

2. Creo View image normalization

Because Creo View introduces:
* glare
* reflections
* shadows
* uneven lighting
  the program preprocesses the image before mask extraction.

This normalization step:
* reduces harsh reflections
* compresses brightness variation
* preserves edges better than a normal blur
* makes magenta and cyan highlight colors more stable

3. Two-mask extraction

The system extracts two different masks:

Focus mask
* Represents the main part in focus
  Expected highlight color:
* magenta (255, 0, 255)

Context mask
* Represents the closest related parts
  Expected highlight color:
* cyan (0, 255, 255)

At first, mask extraction was RGB-based, but that failed in shadow.
So now the program mainly uses HSV-based masking, which is much more robust when:
* the highlight is darkened
* the highlight is desaturated by shadow
* the image lighting is uneven

4. Mask cleanup

After the masks are extracted, the program cleans them using morphological operations.

This removes:
* noise
* tiny fragments
* small holes

So the masks become more stable and usable for rendering.

5. User confirms masks

The user can:
* tune mask sensitivity
* regenerate masks
* inspect the result

Only when satisfied does the user continue to rendering.

6. Object vs background detection

Once masks exist, the renderer must figure out:
* what is object
* what is background

This became a major challenge because bright reflections on object surfaces looked like background.
So the program now uses improved background segmentation logic:
* brightness threshold
* low-variance detection
* border-connected background logic

That means:
* only bright regions connected to the image border count as background
* bright reflections inside the object do not become background

7. Color mapping into company illustration standard

Once the masks and object region are known, the renderer converts the image into company target color logic:

Focus part → orange
Closest parts → dark gray
All other visible parts → light gray
Background → white

This is now deterministic.

8. Shading preservation

The program does not just paint flat color.
It uses the source image luminance to preserve some 3D readability:
* the focus part keeps stronger form and detail
* context parts are flatter and quieter

That gives the output a technical-illustration look while still preserving shape.

9. Contour enhancement

The renderer also strengthens important outlines.
This helps the image look more like a service illustration and less like a raw recolored screenshot.

It improves:
* silhouettes
* part readability
* visual clarity of the focus object

10. Annotation workspace

After rendering, the user enters the annotation phase.
There are two main editors:

Line editor
Used to draw movement lines manually

Callout editor
Used to place callout circle center and leader endpoint

11. Manual annotation rendering

User annotations are rendered on top of the illustration.

Movement lines:
are black
now have a white halo
which improves readability against dark geometry

Callouts:
use a white blocking circle
black label
black leader line

This follows company illustration logic much more closely now.

12. AI assistance

AI chooses strategy
The model decides things like:
* task type
* target mode
* direction
* movement-line strategy
* callout strategy

It does not directly own final placement.

13. Focus target analysis

To help AI make better decisions, the program now analyzes the focus mask.
It detects:
* connected components
* component count
* centroids
* bounding boxes
* whether the focus looks like:
* one large part
* repeated small parts
* mixed parts

This means AI no longer reasons from only a vague bounding box.
Instead, it gets structured information about what the highlighted area actually looks like.

14. Strategy-to-annotation conversion

After AI returns a strategy, deterministic code converts that into actual annotations.
This file is one of the key parts of the architecture now.

It takes:
* AI strategy
* focus mask analysis
* image size

and turns that into:
* movement lines
* callouts

So the final annotation placement is rule-driven, not hallucinated.

15. Improved movement-line logic

Movement lines have gone through several upgrades.

They now support:
large part = 1 directional line
repeated small parts = up to 3 directional lines

For repeated small parts, the system no longer draws three parallel lines from one centroid.
Instead it:
* selects spread-out target components
* places one line per selected component
* keeps direction and length consistent

This made the result much closer to real service-illustration behavior.

16. Collision-aware line placement

The latest improvement is collision-aware placement.
The system now tries several candidate line positions and scores them based on:
* overlap with the focus mask
* closeness to already placed AI lines
* unnecessary shift amount

Then it picks the cleaner option.
This is still a lightweight heuristic, but it already makes AI movement lines look much more professional.

17. Pending vs approved AI suggestions

AI suggestions are not automatically final.
The program distinguishes between:

Pending AI suggestions
Shown in preview only
Styled differently so the user can review them

Approved AI lines
Explicitly accepted by the user
Included in final export

18. Export pipeline

The final rendered image can be exported in multiple formats:
GRIPS:
* TIFF
* JPEG
  SID:
* TIFF
* JPEG
  Regular:
* PNG
* JPG

The export logic:
* resizes correctly
* preserves aspect ratio
* pads icons to square when needed
* sets DPI metadata
* applies TIFF/JPEG output settings