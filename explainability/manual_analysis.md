# Manual Disagreement Source Analysis

This compulsory review inspects the 20 highest-entropy CIFAR-10H test images and assigns one primary source of human disagreement to each image. The examples are selected by sorting Shannon entropy over the full CIFAR-10H soft-label matrix in descending order.

Categories used in the table:

- `ambiguous identity`: the central object plausibly resembles more than one CIFAR-10 class.
- `poor quality`: blur, tiny object size, occlusion, low contrast, or compression makes recognition difficult.
- `multi-object`: multiple objects or strong contextual distractors compete for the annotator's attention.
- `boundary case`: the image sits near a semantic boundary between related classes or between foreground/background cues.
- `other`: disagreement source does not fit the categories above.

## High-Entropy Review Table

| Entropy rank | CIFAR-10 test index | Entropy (bits) | Original label | Top human labels | Manual source category | Manual note |
|---:|---:|---:|---|---|---|---|
| 1 | 6750 | 2.86 | deer | frog 0.25, dog 0.17, deer 0.17, cat 0.12 | poor quality | Small low-resolution animal in vegetation; the body outline is not clear enough to separate deer/frog/dog/cat confidently. |
| 2 | 8153 | 2.43 | deer | deer 0.38, dog 0.20, cat 0.16, bird 0.08 | poor quality | Object is tiny and low-contrast against a pale background, so annotators rely on weak shape cues. |
| 3 | 6792 | 2.37 | cat | cat 0.32, truck 0.22, ship 0.22, dog 0.12 | boundary case | Dark blocky foreground and background geometry create a boundary between animal and vehicle/ship-like silhouettes. |
| 4 | 86 | 2.32 | bird | bird 0.48, ship 0.15, frog 0.12, dog 0.10 | poor quality | Heavy blur and partial visibility make the central object difficult to localize. |
| 5 | 2232 | 2.27 | airplane | airplane 0.47, cat 0.18, bird 0.16, dog 0.06 | ambiguous identity | The dark angular foreground can be read as aircraft structure, bird shape, or animal head depending on the visual cue chosen. |
| 6 | 5840 | 2.26 | bird | bird 0.56, horse 0.10, cat 0.10, dog 0.08 | poor quality | Motion blur and low contrast obscure the object boundary. |
| 7 | 3463 | 2.25 | cat | cat 0.41, dog 0.29, bird 0.12, automobile 0.07 | multi-object | Foreground animal and surrounding clutter split attention between cat/dog-like cues and background objects. |
| 8 | 5369 | 2.22 | deer | deer 0.38, bird 0.24, horse 0.16, dog 0.14 | ambiguous identity | Animal pose and texture overlap with several animal classes, especially deer, bird, horse, and dog. |
| 9 | 6197 | 2.21 | deer | bird 0.42, deer 0.24, dog 0.12, horse 0.08 | ambiguous identity | The upright animal shape is visually compatible with bird and deer interpretations. |
| 10 | 5227 | 2.19 | deer | deer 0.47, horse 0.14, frog 0.14, dog 0.12 | ambiguous identity | Fine-grained animal identity is unclear because the object is small and partly blended into foliage. |
| 11 | 3391 | 2.19 | cat | cat 0.34, dog 0.28, bird 0.26, deer 0.04 | boundary case | The animal is stretched/partly occluded, giving annotators enough cues for cat, dog, or bird-like reads. |
| 12 | 4821 | 2.18 | deer | deer 0.44, cat 0.17, bird 0.13, frog 0.12 | poor quality | Green/white blur dominates the crop, reducing the usable object signal. |
| 13 | 8855 | 2.18 | deer | deer 0.44, cat 0.18, bird 0.14, frog 0.12 | poor quality | Blurry object and background vegetation make the class boundary weak. |
| 14 | 5734 | 2.16 | frog | frog 0.44, bird 0.23, deer 0.13, dog 0.12 | boundary case | Animal is camouflaged in vegetation, creating a boundary between frog-like texture and bird/deer shape cues. |
| 15 | 3357 | 2.14 | ship | cat 0.42, ship 0.30, bird 0.12, frog 0.06 | multi-object | Water/vehicle-like context competes with a foreground shape that annotators often read as an animal. |
| 16 | 7238 | 2.13 | deer | deer 0.41, frog 0.24, bird 0.19, dog 0.09 | boundary case | Thin foreground structures and sand-colored background produce weak evidence for both animal and non-animal readings. |
| 17 | 2855 | 2.12 | cat | cat 0.38, bird 0.25, frog 0.23, dog 0.09 | poor quality | Close-up crop is occluded and lacks enough global shape to settle the class. |
| 18 | 5837 | 2.11 | bird | bird 0.48, cat 0.20, dog 0.16, frog 0.06 | multi-object | Dark foreground, vegetation, and possible multiple subjects create competing visual explanations. |
| 19 | 6024 | 2.10 | bird | bird 0.62, truck 0.08, deer 0.06, automobile 0.06 | boundary case | The vertical central shape and background geometry add vehicle-like alternatives despite bird being the strongest label. |
| 20 | 3113 | 2.08 | cat | bird 0.33, cat 0.29, frog 0.27, horse 0.04 | ambiguous identity | Curved dark shape and low visibility make the object plausible as bird, cat, or frog. |

## Summary For Viva

The dominant disagreement sources are poor image quality and ambiguous animal identity. The highest-entropy cases often combine tiny object size, blur, occlusion, and animal-class overlap, which explains why a soft distribution is more faithful than a single hard label.

Category counts:

| Category | Count |
|---|---:|
| poor quality | 7 |
| ambiguous identity | 5 |
| boundary case | 5 |
| multi-object | 3 |
| other | 0 |

To reproduce the candidate set, run `python data/dataset.py` after downloading data and sort CIFAR-10H examples by Shannon entropy descending. The row IDs above use the canonical CIFAR-10 test-set index, so they remain stable under the deterministic project seed.
