#include "f_statistic.h"
#include "f_statistic_reference.h"

/* Compile-only smoke test: the public header and the reference header must
 * be consumable from plain C11, and every prototype must be the one the
 * kernel core 3.0.0 ABI declares. Nothing here runs. */
void test_fstat_c_header(void)
{
    int detector_window_samples = 0;
    int num_weight_terms = 0;
    int sample_bits_per_component = 0;
    int reference_offset_bins = 0;
    int windows_per_stream = 0;
    int pad_factor = 0;
    int fine_bins = 0;
    unsigned long long fine[FSTAT_NUM_WEIGHT_TERMS][FSTAT_FINE_NUM_BINS];
    unsigned long long powers[FSTAT_NUM_WEIGHT_TERMS];

    (void)sizeof(InputType);
    FStat_GetSpecs(
        &detector_window_samples,
        &num_weight_terms,
        &sample_bits_per_component,
        &reference_offset_bins);
    FStat_GetVersion(0, 0, 0);
    FStat_GetFineSpecs(&windows_per_stream, &pad_factor, &fine_bins);
    (void)FStat_Create(0, 1);
    (void)FStat_Create_Batch(0, 1, 1);
    (void)FStat_Supports_FusedFine();
    FStat_Compute_FusedFine_U64(0, 0, 0, 0, 0);
    (void)FStat_Supports_FusedFineMaskWithSupport();
    FStat_Compute_FusedFineMaskWithSupport_U64(
        0, 0, 0, 0, 0, 0, 1ULL, 0, 0, 0, 0, 0, 0);
    fstat_ref_fine_powers_u64(0, 0, fine);
    fstat_ref_powers_from_fine_u64(fine, powers);
}
