# Data

This directory holds ADNI data files used for the empirical validation in Section 3 of the paper. These files are not included in the repository — ADNI data is distributed under a data use agreement and cannot be redistributed.

## Accessing ADNI Data

1. Register for access at https://adni.loni.usc.edu/
2. Once approved, log in to the LONI Image and Data Archive (IDA)
3. Navigate to **Download → Study Data → Key ADNI tables**
4. Download the following two files:
   - `DXSUM_PDXCONV_ADNIALL.csv` — longitudinal diagnosis table
   - `NEUROBAT.csv` — neuropsychological battery (RAVLT, BNT, Digit Span)
5. Rename both files to match the pattern `DXSUM_*.csv` and `NEUROBAT_*.csv` and place them in this directory

## Files Used

| File | Contents | Used for |
|---|---|---|
| `DXSUM_*.csv` | Visit-level diagnosis codes (1=CN, 2=MCI, 3=Dementia) | Identifying MCI→AD converters and conversion timing |
| `NEUROBAT_*.csv` | RAVLT delayed recall, Boston Naming Test, Digit Span scores | Longitudinal cognitive trajectories per domain |

## Cohort Definition

The analysis selects subjects who:
- Had a baseline diagnosis of MCI (DIAGNOSIS = 2.0)
- Later converted to AD/Dementia (DIAGNOSIS = 3.0) at any follow-up visit
- Had cognitive scores measured within a ±36-month window around conversion
- Had at least two valid observations per cognitive domain

The resulting cohort and per-subject slope estimates are saved to `week6_adni_cohort.csv` and `week6_failure_sequence.csv` (these output files are safe to commit and are included in `results/`).
