# Measured results

The committed result sets were produced by sequential Hugging Face Jobs using one 96 GB RTX PRO
6000 Blackwell at a time, within one cumulative three-hour/$10 envelope. Raw server logs,
telemetry, quality gates, and machine-readable benchmark output are retained here.

- `mtp-partial`: initial kernel/chunk sweep.
- `dspark-sweep`: trained DSpark/DFlash-family draft sweep.
- `mtp-depth-sweep`: native MTP verification depths 2 through 7.
- `final-validated`: clean finalist reproduction, exact 262,080-token proof, and stability soak.
- `advanced-partial`: decode-mode/Spec-V2 comparison before the short-job guard was adjusted.
- `advanced-final`: completed decode-mode/Spec-V2/page-size sweep and final 39-cycle soak.

`config/selected_profiles.env` contains the fastest quality-safe, full-context profiles.
