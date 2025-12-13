# Changelog

All notable changes to this project are documented here.  
This file is created to track improvements made based on peer review feedback and other updates to the project.

---

## [Unreleased] - 2025-12-13

### Addressed Peer Review Feedback

#### Issue #38 (Reviewer: Daniel Yorke)

https://github.com/UBC-MDS/data-analysis-review-2025/issues/38

- **Change:** Suppressed warnings in scripts outputs
  **Who:** @Purityj  
  **Evidence:** PR: 

 **Change:** Replaced "samples" with "observations" throughout report and documentation.  
  **Who:** @luisalonso8  
  **Evidence:** See PR [75](https://github.com/junliliu1/wine_quality_predictor/pull/75) and notebook outputs in `reports/wine_quality_predictor_report.qmd`.

  - **Docker build performance improved:** Built docker before pushing to Dockerhub so that next time a user starts the container they can just use `docker compose up`  
  **Who:** @Purityj  
  **Evidence:** PR: https://github.com/junliliu1/wine_quality_predictor/pull/78


#### Issue #82 (Reviewer: Manikanth Goudimani)

https://github.com/UBC-MDS/data-analysis-review-2025/issues/82

- **Change:** Updated the Discussion section based on reviewer comments.  
  **Who:** @luisalonso8  
  **Evidence:** PR [#77](https://github.com/junliliu1/wine_quality_predictor/pull/77), Discussion section and after conclussion in `reports/wine_quality_predictor_report.qmd`.

#### Issue #57 (Reviewer: Sam Lokanc)

https://github.com/UBC-MDS/data-analysis-review-2025/issues/57

- **Change:** Converted classification report into a table format.  
  **Who:** @jimmy2026-V  
  **Evidence:** PR: [#68] (https://github.com/junliliu1/wine_quality_predictor/issues/68)

- **Change:** Updated Discussion section.  
  **Who:** @luisalonso8  
  **Evidence:** PR [#77](https://github.com/junliliu1/wine_quality_predictor/pull/77), Discussion Section  in `reports/wine_quality_predictor_report.qmd`.

#### Issue #49 (Reviewer: Hugo Kwok)

https://github.com/UBC-MDS/data-analysis-review-2025/issues/49

- **Change:** Model comparison added to Discussion section.  
  **Who:** @luisalonso8  
  **Evidence:** PR [#77](https://github.com/UBC-MDS/data-analysis-review-2025/issues/49), Discussion Section and after conclussion in `reports/wine_quality_predictor_report.qmd` in Discussion section .
- **Change:** Notebook moved to `analysis/` folder.  
  **Who:** @jimmy2026-V  
  **Evidence:** See PR [#79](https://github.com/junliliu1/wine_quality_predictor/pull/79).  
  
- **Change:** PDF rendering from HTML fixed.  
  **Who:** @junliliu1  
  **Evidence:** PR: 

  - **Change:** HTML rendering Improved - all images are rendering correctly
  **Who:** @Purityj  
  **Evidence:** PR: https://github.com/junliliu1/wine_quality_predictor/pull/67 and https://github.com/junliliu1/wine_quality_predictor/pull/72

### Other Changes
- **README.md edits to incorporate Milestone 4 changes.**  
  **Who:** @Purityj  
  **Evidence:** PR: https://github.com/junliliu1/wine_quality_predictor/pull/76

---

### [3.0.0] - 2025-12-13
### Added peer-review fixes
- Suppressed warnings in outputs
- Fixed PDF rendering issue
- Updated Discussion section based on feedback
- Fixed html rendering issues 
- Converted classification report into a table format
