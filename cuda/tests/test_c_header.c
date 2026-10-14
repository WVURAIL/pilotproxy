#include "f_statistic.h"

void test_fstat_c_header(void)
{
    FStatRational r;
    int detector_window_samples = 0;
    int num_weight_terms = 0;
    int sample_bits_per_component = 0;
    int reference_offset_bins = 0;

    (void)sizeof(InputType);
    (void)FStat_NumDenToRawF(1ULL, 2ULL);
    (void)FStat_NumDenToPilotExcess(1ULL, 2ULL);
    (void)FStat_NumDenToPilotExcessChecked(1ULL, 2ULL, 0);
    (void)FStat_MakeHalfThresholdFromFullChecked(1ULL, 2ULL, &r);
    FStat_GetSpecs(
        &detector_window_samples,
        &num_weight_terms,
        &sample_bits_per_component,
        &reference_offset_bins);
    FStat_GetVersion(0, 0, 0);
    (void)FStat_Supports_FusedFineMaskWithSupport();
    FStat_Compute_FusedFineMaskWithSupport_U64(
        0, 0, 0, 0, 0, 0, 1ULL, 0, 0, 0, 0, 0, 0);
}
