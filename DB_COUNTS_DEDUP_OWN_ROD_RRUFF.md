# Raman DB Counts Without Doubled Entries

Count date: 2026-03-21

## Scope Used by App

Loaded from `MATCH_FOLDERS_STANDARD` via `raman_core.load_reference_folders(...)`.

- OWN: `databases/OWN`
- ROD: `databases/ROD`
- RRUFF: `databases/RRUFF`

- Loader skips: `0`

## Dedup Rule Used (Primary)

Primary deduplication = **one entry per mineral/phase name (case-insensitive) within each source DB**.

## Counts (Primary Dedup: by name)

| Source | Raw spectra entries | Deduplicated entries (unique names) |
|---|---:|---:|
| OWN | 8 | 5 |
| ROD | 1133 | 408 |
| RRUFF | 38056 | 2511 |
| **Total** | **39197** | **2924** |

## Class Counts (Primary Dedup: by name)

| Class | OWN | ROD | RRUFF | Total |
|---|---:|---:|---:|---:|
| Arsenates | 0 | 0 | 153 | 153 |
| Borates | 0 | 0 | 34 | 34 |
| Carbonates | 0 | 3 | 158 | 161 |
| Halides | 0 | 13 | 40 | 53 |
| Molybdates | 0 | 1 | 8 | 9 |
| Nitrates | 0 | 1 | 12 | 13 |
| Other classes | 2 | 8 | 86 | 96 |
| Oxides / hydroxides | 3 | 260 | 496 | 759 |
| Phosphates | 0 | 0 | 267 | 267 |
| Silicates | 0 | 105 | 754 | 859 |
| Sulfates | 0 | 0 | 214 | 214 |
| Sulfides | 0 | 17 | 203 | 220 |
| Tungstates | 0 | 0 | 5 | 5 |
| Unknown formula | 0 | 0 | 51 | 51 |
| Vanadates | 0 | 0 | 30 | 30 |
| **Class total check** | **5** | **408** | **2511** | **2924** |

Requested key classes (primary dedup):
- Oxides / hydroxides: **759**
- Silicates: **859**
- Sulfides: **220**

## Alternative Strict Dedup (by name + formula)

| Source | Deduplicated entries (unique name+formula) |
|---|---:|
| OWN | 5 |
| ROD | 446 |
| RRUFF | 2511 |
| **Total** | **2962** |

## Output Lists

- `DB_LIST_DEDUP_BY_NAME.csv` (primary deduplicated list)
- `DB_LIST_DEDUP_BY_NAME_FORMULA.csv` (strict deduplicated list)
