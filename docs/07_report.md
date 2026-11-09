# The LaTeX report

The report lives in `report/` and is compiled on **Overleaf** (no TeX is installed locally).

## Files

| File | Content | Source |
|---|---|---|
| `main.tex` | title, authors, section order, packages | — |
| `sections/00_abstract.tex` … `10_conclusion.tex` | one file per section | ported from the proposal doc |
| `sections/07_results.tex` | results tables + figures | filled from `results/*.csv` and `figures/` after P6–P7 |
| `references.bib` | the 11 references from the proposal | check authors marked `TODO` |
| `figures/` | images the report includes | copied from `../figures/` |

Anything still missing is marked `\todo{...}`, which prints in **red** in the PDF. Before submission,
search for `\todo` — there must be none left.

One file per section means the three of us can edit different sections at the same time without
conflicts.

## Overleaf setup (once)

1. Zip the `report/` folder.
2. Overleaf → New Project → Upload Project → the zip.
3. Menu → Compiler: **pdfLaTeX**. Main document: `main.tex`.
4. Share → add the other two teammates as editors.

## Keeping Overleaf and this folder in sync

Overleaf is the source of truth for **text**. This folder is the source of truth for **figures and result
numbers**. When new figures are made: upload them to Overleaf's `figures/` folder (drag and drop). When text
is edited on Overleaf: Menu → Download → Source, and copy `sections/` back here before committing.

(Overleaf's paid plan has git sync; the free plan needs the manual steps above.)

## What changed from the proposal doc

- "We propose / we will" wording in the abstract changed to past tense where it describes the work.
- Section 6 (evaluation plan) became **Experimental setup**; a new **Results** section was added.
- Section 7 (showcase) became **Live demo**, trimmed to what the demo shows.
- Section 8 (timeline and risks) was dropped: it is planning material, not report content. Risks that
  actually happened can go in Limitations (section 10).
- The pipeline diagram is redrawn in TikZ; the Colab 3B row was removed from the models table (skipped
  unless time is left).
