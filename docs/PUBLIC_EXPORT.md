# Public portfolio export

This is a clean source export of the AIC211 group academic workspace. It starts a new Git history rather than publishing the original workspace's objects.

## Included

- The original `app.py`, Python modules in `src/`, research configuration metadata, and research dependency locks.
- Original research tests in `tests/` as reviewable source; these are not the public CI target.
- Selected research utilities, the original local launcher, and loopback Streamlit configuration.
- A separate synthetic-only demo, two fabricated email fixtures, and focused public-clone checks.

## Excluded

- Virtual environments, `.git`, local agent metadata, and machine-specific settings.
- Email corpora, real messages, model weights, research caches/predictions, feedback databases, logs, and archives.
- Manuscripts and rendered papers, which can contain author/student identifiers or historical results.
- The original production environment templates and machine-specific PowerShell/model-path scripts.

The dataset registry is retained for provenance review. Some entries are explicitly excluded or have unresolved terms; listing them does not make their contents part of this export.

## Runnable scope

`tools/demo_indicators.py` executes the existing `src.security_matrix.build_12_layer_matrix` on synthetic email. It does not load a trained model, call an LLM, make a final classification, or open email URLs. Layers with unavailable evidence remain `UNKNOWN`.

The full dashboard needs compatible trusted model bundles and, where applicable, separately provisioned research data and a local LLM. No public-clone claim is made that the full research test suite or a trained model runs without those assets.

## Attribution and reporting

The source is a team academic project, not solely the work of the account publishing this copy. No new project-wide license is assigned here. AI assistance was used in parts of the project. Quantitative claims must be tied to an identified run and evaluation protocol; historical headline metrics have deliberately not been promoted in the portfolio README.

## Validation

The local preparation checks compile the published Python source, execute the synthetic offline demo and the tests in `tests_public/`, inspect the publication file list, and scan included text for common credential patterns. This is a bounded publication review, not a complete security audit, a verification of the original Git history, or a reproduction of the research benchmark.
