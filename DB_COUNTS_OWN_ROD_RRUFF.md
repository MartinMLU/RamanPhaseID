# Raman DB Counts (OWN, ROD, RRUFF)

Count date: 2026-03-21

## Scope Used by App

Counts are based on `MATCH_FOLDERS` in `RamanPhaseID_0p99beta.py` and loaded via the app's database inventory path (the same path used by the sidebar cache overview).

- OWN: `databases/OWN`
- ROD: `databases/ROD`
- RRUFF: `databases/RRUFF`

No loader skips were reported (`skipped = 0`).

## Core Counts

| Source | Spectra entries | Different minerals/phases (case-insensitive) |
|---|---:|---:|
| OWN | 8 | 5 |
| ROD | 1133 | 408 |
| RRUFF | 38056 | 2511 |
| **Total (OWN+ROD+RRUFF)** | **39197** | **2618** |

## Class Counts (By Spectra Entries)

Classification is formula-based, single-label, precedence order:
`Silicates > Carbonates > Phosphates > Arsenates > Vanadates > Molybdates > Tungstates > Borates > Nitrates > Sulfates > Halides > Sulfides > Oxides/Hydroxides > Other > Unknown`

| Class | OWN | ROD | RRUFF | Total |
|---|---:|---:|---:|---:|
| Arsenates | 0 | 0 | 1608 | 1608 |
| Borates | 0 | 0 | 319 | 319 |
| Carbonates | 0 | 8 | 2553 | 2561 |
| Halides | 0 | 30 | 392 | 422 |
| Molybdates | 0 | 4 | 96 | 100 |
| Nitrates | 0 | 4 | 98 | 102 |
| Other classes | 2 | 13 | 874 | 889 |
| Oxides / hydroxides | 6 | 723 | 5642 | 6371 |
| Phosphates | 0 | 0 | 3969 | 3969 |
| Silicates | 0 | 316 | 16663 | 16979 |
| Sulfates | 0 | 0 | 2249 | 2249 |
| Sulfides | 0 | 35 | 2501 | 2536 |
| Tungstates | 0 | 0 | 312 | 312 |
| Unknown formula | 0 | 0 | 406 | 406 |
| Vanadates | 0 | 0 | 374 | 374 |
| **Class total check** | **8** | **1133** | **38056** | **39197** |

## Requested Key Classes (Totals)

- Oxides / hydroxides: **6371**
- Silicates: **16979**
- Sulfides: **2536**
