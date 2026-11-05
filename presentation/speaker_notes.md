# LunaPSR — speaker notes

**Deck:** `presentation/lunapsr_6slide.pptx` (primary) / `presentation/lunapsr_6slide.html` (browser)  
**Format:** 6 slides, 16:9  
**Target time:** 6–7 minutes (about 60–70 seconds per slide)  
**Audience:** mixed technical / science / demo audience — assume no prior LunaPSR knowledge

> Core framing: LunaPSR makes weak OHRC signal **analyzable**, not “recovered.”
> Every enhanced image is a derived representation. The noise estimate is
> model-based, not measured from the scene.

---

## Slide 1 — Title: “LunaPSR”

**Goal:** establish the problem and the honesty constraint in one breath.

**Say:**

> Permanently shadowed regions near the lunar poles are some of the most
> scientifically interesting places on the Moon — and some of the hardest to
> image. Chandrayaan-2’s OHRC sees them at only a handful of digital numbers
> above the noise floor.
>
> LunaPSR is a read-only enhancement and analysis system for that signal. In
> plain terms: it takes a very faint lunar image, makes it easier to study,
> compares enhancement methods, and records exactly what each step changed.
> We are not trying to invent detail the camera did not capture.

**Point at the three at-a-glance cards (PPTX) or the three stats (browser/PDF):**

- **What it is** — a software lab for real Chandrayaan-2 OHRC lunar images.
- **What it does** — enhances, compares, measures, and records every derived step.
- **Why it matters** — PSRs are science-rich, but only a few DN above the noise.

If you are using the browser/PDF version, the stats are **1–5 DN** (source signal),
**0.511 DN** (modeled noise), and **1.2 Gpx** (product scale).

**Transition:**

> That gap between “a few DN” and “noise” is the whole project. Let’s look at
> why it is hard.

---

## Slide 2 — Problems it solves: “The signal is barely above the noise”

**Goal:** make the audience feel the signal-to-noise problem and the danger of
over-enhancement — then name the problems the project addresses.

**Say:**

> In the studied region, raw values span about 1 to 5 DN, with a mean of 3.056
> and a standard deviation of 0.467. The Phase 4B model puts total noise at
> roughly 0.511 DN. That gives a **model-based SNR of about 6** — not a measured
> SNR. It is a prediction from the detector model.
>
> A global stretch mostly makes the noise visible. Push a Retinex-style
> enhancement and it can look convincing while adding structure that was not in
> the data. In a 5-DN ROI, standard SSR produced **31.4% new edges**, added
> **391 isolated components**, and strengthened only **40.7%** of the original
> edges.
>
> So the question we care about is not “does it look better?” It is “is any of
> it real?”
>
> That is why the project solves five practical problems: weak contrast, noise
> amplification, false detail, no ground truth, and no audit trail.

**Transition:**

> To answer that, we built a pipeline that keeps the source data untouched and
> makes every derived step auditable.

---

## Slide 3 — What it does + features: “One workflow: from raw pixels to evidence you can review”

**Goal:** show that this is a real system, not a single filter — and make the
features concrete.

**Say:**

> The pipeline starts with the real OHRC PDS4 product. PDS4 is the mission’s
> standard data format. We memory-map the 1.2-gigapixel array, mask invalid
> pixels, normalize brightness reproducibly, and run a physics-based noise
> model. Only after that do we enhance: classical baselines, LunaPSR, and
> Adaptive Retinex.
>
> Every output is a derived representation under `outputs/`. The source array is
> never modified. The app records diagnostics: new-edge fraction, edge
> preservation index, noise amplification, and the parameters used.
>
> The six feature cards are the practical capabilities: read-only handling,
> noise awareness, multiple methods, diagnostics, auditability, and a usable
> 9-tab Streamlit lab with **304 tests**.

**Transition:**

> The key design choice is that the noise model is not a post-hoc excuse. It is
> the gate.

---

## Slide 4 — How it works: “Enhance only where the signal can support it”

**Goal:** explain the noise-aware gate and the Adaptive Retinex prototype
without overstating Phase 5B.

**Say:**

> The Phase 4B model combines shot noise, read noise, and quantization noise.
> Gain is inferred from the product metadata; we do not claim it is an official
> calibration value.
>
> From that model we build a two-factor confidence: how strong the local signal
> is, and whether local variation rises above what the noise model predicts.
> The adaptive strength α stays conservative where the variation is consistent
> with noise.
>
> Adaptive Retinex makes the foundation explicit: an illumination estimate, a
> robust log-Retinex response, and a medium-scale structure map. It is not one
> global curve applied to the whole image.
>
> Phase 5B is still a prototype — validation across regions of interest is
> ongoing.

**Transition:**

> So does it actually help? We built a ground-truth benchmark to answer that.

---

## Slide 5 — Evidence: “We tested it against known ground truth — and report where it loses”

**Goal:** deliver the strongest evidence and the honest limitations. This is
the credibility slide; do not rush it.

**Say:**

> Phase 5C takes eight real 256×256 OHRC dark tiles — real noise, real
> quantization, real background statistics — and injects craters and boulders
> at known positions and amplitudes. We run five methods through one generic
> detector, with a per-tile 5% false-positive calibration.
>
> The headline is not a clean sweep. **Raw is the best overall detector**, with
> an AUC of 0.882. LunaPSR is 0.868. At or below 0.5 DN, LunaPSR ties raw
> overall — 0.744 versus 0.742 — and it wins on faint 12–20 pixel boulders:
> 0.704 versus 0.638 for raw, and 0.559 for SSR.
>
> But we also report where it loses. LunaPSR is the worst method for 6-pixel
> boulders (7.84 σ at 80% detection), and it is less sensitive than raw for faint
> craters. SSR amplifies noise by about 50× on control tiles; LunaPSR is about
> 14.5×.
>
> The win is narrow, measurable, and honest. That is more useful than claiming
> a clean sweep.

**Transition:**

> That is the evidence. The last slide is what you can try, and what we would
> do next.

---

## Slide 6 — Use it: “What you can do with LunaPSR”

**Goal:** make the ask concrete and close on the honest value proposition.

**Say:**

> You can inspect a real OHRC product read-only, compare gamma, CLAHE, SSR,
> denoisers, LunaPSR, and Adaptive Retinex side by side, see confidence, noise,
> structure, and edge diagnostics, and export a reproducible run record.
>
> The live demo is currently private; the owner can switch visibility to public.
> Cloud runs use the bundled sample strip by design. The full 1.2 Gpx product is
> not in the repository, so local runs are where the complete product and the
> lab are exercised.
>
> Current status: Phase 5C is complete, Phase 5B Adaptive Retinex is still a
> prototype, and the live demo is private. What we want next: validate across
> more regions, add a downstream detection task, publish one guided real-tile
> before/after, and make the demo public.
>
> The goal is not a prettier image. It is a more trustworthy analysis — knowing
> when enhancement helps, when it does not, and when it is only amplifying noise.

**Close:**

> Thank you. I’m happy to go deeper on the noise model, the Phase 5C design, or
> the app.

---

## Q&A guardrails

Use these short answers to keep the claims precise.

**“Did you recover hidden geology?”**  
No. We produce derived representations. Phase 5C uses injected ground truth to
measure detection, not to prove that a real enhanced structure is geological.

**“Is the SNR measured?”**  
No. It is **model-based**, from the Phase 4B detector noise model. The deck
labels it that way deliberately.

**“Why does raw win overall?”**  
Because the controlled features are detectable without enhancement in many
cells, and enhancement can hurt detection. LunaPSR’s measured advantage is
narrow: the faint, low-contrast, 12–20 pixel boulder regime.

**“Is this super-resolution or AI?”**  
No. There are no neural nets or generative models in the core pipeline. It is
classical enhancement plus a physics-based noise model and diagnostics. No
method can create photons the sensor did not collect.

**“Why not deep learning?”**  
That would need a trustworthy training/validation set and a much stronger
ground-truth story. The current benchmark is deliberately controlled and
classical so the claims stay falsifiable.

**“What does ‘new-edge fraction’ mean?”**  
The fraction of edges in the enhanced output that were not present in the
original. It is one diagnostic of manufactured structure, not proof that every
new edge is wrong.

---

## Demo choreography (optional, if live)

1. Open the app and show the product/validation view. Point out the read-only
   source and the warning copy.
2. Switch to Enhancement. Show the sample strip and a conservative
   gamma/CLAHE result. Point out the scientific warnings.
3. Open the OHRC Lab. Show the parameter documentation and the Original / SSR /
   LunaPSR views. If running locally on the full product, use a crop and keep
   the interaction short.
4. Open Metrics or Noise. Show the model-based noise estimate and the
   diagnostics.
5. Open History. Show that a run is recorded with its parameters.
6. Close on the limitation: the app is an analysis tool, not a geological
   discovery claim.

---

## Export / presenting

- Primary: open `presentation/lunapsr_6slide.pptx` in PowerPoint. The speaker
  notes are embedded.
- Browser: open `presentation/lunapsr_6slide.html`; use **← / →** or **Space**
  to navigate, **F** for fullscreen.
- To export a PDF from the browser: **Ctrl/Cmd+P** → landscape → margins
  **None** → enable **Background graphics** → save as PDF.
- Keep the `presentation/` folder in place: the HTML references
  `../screenshots/`, `../outputs/phase5c/`, and the OHRC browse image with
  relative paths.
