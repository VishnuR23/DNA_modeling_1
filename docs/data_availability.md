# Dataset availability audit

Checked 2026-09-10 using public, unauthenticated HTTPS endpoints. No request forms, logins, or contact with authors were used. This audit verifies public manifests and representative trajectory byte access, **not full-file integrity or successful preprocessing**.

## Timewarp: public and anonymously downloadable

Official sources: [Microsoft repository](https://github.com/microsoft/timewarp), [dataset card](https://huggingface.co/datasets/microsoft/timewarp), and [public file tree](https://huggingface.co/datasets/microsoft/timewarp/tree/main). The dataset card declares MIT licensing. Keep its attribution and license metadata with any downloaded data.

The recursive API tree was followed through all pagination links for `4AA-large` and `4AA-huge`. The exact results, including file sizes and object/LFS hashes, are archived in [large manifest](sources/4AA-large-manifest.json) and [huge manifest](sources/4AA-huge-manifest.json). “Complete pair” means both `<sequence>-traj-state0.pdb` and `<sequence>-traj-arrays.npz` exist; it does not yet mean the pair passes chemical or timestamp validation.

| Public directory | PDB files | NPZ files | Complete pairs | Total listed bytes, decimal GB |
|---|---:|---:|---:|---:|
| 4AA-large/train | 1459 | 1457 | **1457** | 157.58 |
| 4AA-large/val | 379 | 368 | **368** | 393.94 |
| 4AA-large/test | 182 | 164 | **164** | 165.89 |
| 4AA-huge/train | 1 | 1 | **1** | 2.55 |
| 4AA-huge/val | 119 | 85 | **81** | 171.17 |
| 4AA-huge/test | 96 | 96 | **96** | 202.12 |

The whole large directory is 717.42 GB; huge is 375.83 GB. The README's nominal 1500/400/433 split and TW code's historical inventory numbers are not the present downloadable inventory. TITO's 1457 train count matches complete large training pairs; its 92 huge test molecules do not match today's 96 complete pairs. The exact four exclusions cannot be recovered from the inspected paper/code. They must not be chosen by which generated results look best.

Missing NPZs in large/train are **KTYK, VMRH**. The complete manifests preserve the remaining missing-file evidence. Huge/val has four NPZ-only names (CWVY, GKNV, KRDT, SNLS) and 38 PDB-only names. Do not silently borrow a topology from another split or treat an incomplete pair as valid.

Small access probes:

- [large/test/AAAY trajectory](https://huggingface.co/datasets/microsoft/timewarp/resolve/main/4AA-large/test/AAAY-traj-arrays.npz): HTTP **206**, 1,024 bytes fetched with `Range: bytes=0-1023`.
- [huge/test/AAEW trajectory](https://huggingface.co/datasets/microsoft/timewarp/resolve/main/4AA-huge/test/AAEW-traj-arrays.npz): HTTP **206**, 1,024 bytes fetched identically.

No auth token was supplied. These confirm trajectory bytes are accessible, not just a public README. Full dataset downloads are deferred to Phase 2. A Python HTTPS attempt encountered the local interpreter's missing CA trust configuration; retrying with system curl succeeded, without disabling certificate verification.

Split policy proposed for Phase 2:

1. Preserve published directory membership. Draw the training subset only from the 1457 complete large/train pairs using a logged seed and save the exact names before training.
2. Keep **all 368 available large validation pairs** for validation and all **164 large test + 96 huge test pairs** for held-out evaluation. Do not subsample these molecular sets for a better score. Fewer generated samples per molecule is a separately disclosed compute reduction.
3. Use large validation only for model/threshold selection; huge/test is the main TITO-comparison set. Huge/val and its one training trajectory are not needed and are not pooled into training. Availability does not make every directory part of the same experiment.
4. No sequence-name overlap was found between large/train and huge/test, large/val and huge/test, or large/train and huge/val. Recheck topology/sequence identities after reading full files. Report overlaps between large/test and huge/test as repeated systems with longer references rather than independent molecules.
5. Apply prespecified integrity/force-field checks and record every exclusion, including absent files. If these checks happen to produce 92 huge test molecules, document why; do not force the count to 92.

Raw NPZ trajectories include positions, velocities/forces and other arrays; the model needs positions and exact timestamps. Logarithmic recording around 5 ps anchors means raw array indices are not uniform physical times. Retain the 5 ps anchor grid used by TITO, validating timestamps rather than assuming `frame*1 ps`. Stream each molecule through preprocessing, retain the public source checksum and compact position/time arrays, and bound local disk use. Do not load the whole corpus into memory.

## MDQM9-nc: public; the fallback-to-peptides condition is not met

The original dataset is [Zenodo record 10579242](https://zenodo.org/records/10579242), with an [official loader repository](https://github.com/olsson-group/mdqm9-nc-loaders). Its API explicitly reports `access_right=open`, license `cc-by-4.0`. Archived metadata: [mdqm9-record.json](sources/mdqm9-record.json).

It contains ten binary parts `mdqm9-nc_00` through `_09`, each 4,460,858,966 bytes, plus `mdqm9-nc.sdf` (19,981,829 bytes): **44,628,571,489 bytes**, about 44.63 GB. Follow the loader's assembly instructions to obtain the HDF5; do not assume each part is a standalone molecule subset. Split indices are supplied through the official loader repository, not among the eleven Zenodo payload files; their exact inventory needs downloading and validation when this branch is enabled.

A range request to [part 00](https://zenodo.org/api/records/10579242/files/mdqm9-nc_00/content) returned HTTP **206** and 1,024 bytes without credentials. Thus MDQM9-nc is obtainable without a request form. It must not be described as inaccessible or silently removed for access reasons.

The published dataset contains 12,530 noncyclic molecules and an older 100-molecule, 100 ns RE subset. The newer TITO 1 μs, eight-temperature RE campaign and its ultra-long reference ensembles are not established as downloadable by this audit. Access to MDQM9-nc does not establish access to every reference used in TITO's figures.

The source format is HDF5, which the allowed dependency list cannot read directly. **Proposed additional dependency, not installed: `h5py`**, used only at this ingestion boundary. General small-molecule GAFF setup and RDKit chemical featurization would need further dependencies and explicit permission; the peptide work can use NumPy storage and OpenMM/MDTraj topology tools. The 60-hour core proposal prioritizes the phases' peptide tasks; deferring small-molecule training is a budget/scope proposal for review, not the user's inaccessible-data fallback.

## Alanine and excluded data

ITO's loader names three MDShare alanine XTCs and a solute-only topology; see [pinned loader](https://github.com/olsson-group/ito/blob/8310311250e0e3893bc10bbd80ab67d2eab4ae6a/ito/data.py). The named files establish the intended reference, but their anonymous byte access has not been independently verified in this audit; full retrieval is a Phase 1 prerequisite if the paper-comparison track is approved. OpenMM generation is still planned for the requested ff14SB/OBC2 system, explicitly labeled as a different reference ensemble.

DESRES fast-folding data is excluded throughout. No access requests or download attempts were made. The predecessor repository includes a Chignolin demonstration GIF; it is not used as data.
