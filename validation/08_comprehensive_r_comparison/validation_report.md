# Validation: Comprehensive R Comparison (08)

## 1. Test Description
**What is being tested:**
A large-scale parameter sweep comparing `ndimpute` against `NADA::ros` across multiple dimensions:
- Distributions (Lognormal, Normal)
- Sample Sizes (Small, Medium, Large)
- Censoring Levels (Low, Medium, High)
- Detection Limit Types (Single, Multiple)

**Category:**
Tier 4: Comparative Performance (Regression Testing).

## 2. Rationale
**Why this test is important:**
To systematically identify edge cases or parameter combinations where `ndimpute` diverges significantly from the reference standard (R/NADA). This helps pinpoint bugs or limitations in the simplified plotting position logic.

## 3. Success Criteria
**Expected Outcome:**
- **Single Limit:** `ndimpute` should match NADA closely (MAE < 5%).
- **Multiple Limits:** `ndimpute` (using simplified ROS) is expected to diverge from NADA (using Kaplan-Meier ROS). This suite documents *how much* it diverges.
- **Robustness:** No crashes across the ~36 scenarios.

## 4. Execution
1.  Run `Rscript generate_r_suite.R` to generate benchmark CSVs.
2.  Run `python3 validate_suite.py` to process them and generate `suite_results.csv`.

## 5. Results
See `suite_results.csv` for detailed metrics per scenario.
