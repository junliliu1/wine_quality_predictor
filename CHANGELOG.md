# Changelog

All notable changes to this project are documented here.  
This file is created to track improvements made based on peer review feedback and other updates to the project.

---

## [Unreleased] - 2025-12-13

### Addressed Peer Review Feedback

#### Issue #38 (Reviewer: Daniel)

https://github.com/UBC-MDS/data-analysis-review-2025/issues/38

- **Change:** Suppressed warnings in scripts outputs
  **Who:** @Purityj  
  **Evidence:** See PR [abc123](link-to-commit) and notebook outputs in `reports/wine_quality_predictor_report.qmd`.  

- **Change:** Updated the Discussion section based on reviewer comments.  
  **Who:** @luisalonso8  
  **Evidence:** See discussion in Issue #38 [link](link-to-issue) and PR [#42](link-to-PR).

  - **Docker build performance improved:** Built docker before pushing to Dockerhub so that next time a user starts the container they can just use `docker compose up`  
  **Who:** @Purityj  
  **Evidence:** PR [#50](link-to-PR), Dockerfile updated.


#### Issue #82 (Reviewer: Goudimani)

https://github.com/UBC-MDS/data-analysis-review-2025/issues/82

- **Change:** Updated the Discussion section based on reviewer comments.  
  **Who:** @luisalonso8  
  **Evidence:** PR [#45](link-to-PR), lines 120-150 in `reports/wine_quality_predictor_report.qmd`.

#### Issue #57 (Reviewer: Sam)

https://github.com/UBC-MDS/data-analysis-review-2025/issues/57

- **Change:** Converted classification report into a table format.  
  **Who:** @jimmy2026-V  
  **Evidence:** Commit [def456](link-to-commit), Table generated in `reports/wine_quality_predictor_report.qmd`. 

- **Change:** Updated Discussion section.  
  **Who:** @luisalonso8  
  **Evidence:** PR [#47](link-to-PR), lines 180-200.

#### Issue #49 (Reviewer: Hugo)

https://github.com/UBC-MDS/data-analysis-review-2025/issues/49

- **Change:** Model comparison added to Discussion section.  
  **Who:** @luisalonso8  
  **Evidence:** PR [#48](link-to-PR), section “Model Comparison” in `reports/wine_quality_predictor_report.qmd`.  

- **Change:** Notebook moved to `analysis/` folder.  
  **Who:** @jimmy2026-V  
  **Evidence:** See PR [#46](link-to-PR).  
  
- **Change:** PDF rendering fixed from HTML.  
  **Who:** @junliliu1  
  **Evidence:** Commit [ghi789](link-to-commit), `docs/wine_quality_predictor_report.pdf`.

### Other Changes
- **README.md edits to incorporate Milestone 4 changes.**  
  **Who:** @Purityj  
  **Evidence:** PR [#51](link-to-PR), `README.md`.

---

## [1.0.0] - 2025-11-30
### Initial Release
- Implemented Random Forest classifier for wine quality prediction.  
- Generated HTML and PDF reports.  
- Added EDA, model evaluation, and plots.
