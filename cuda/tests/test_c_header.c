#include "pilotproxy_detector.h"
#include "pilotproxy_reference.h"

/* Compile-only smoke test. The public header and the reference header must
 * compile as plain C11, and each call below must match the prototype that
 * the headers declare. Nothing here runs. */
void test_pilotproxy_c_header(void)
{
    int detector_window_samples = 0;
    int num_weight_terms = 0;
    int sample_bits_per_component = 0;
    int reference_offset_bins = 0;
    int windows_per_stream = 0;
    int pad_factor = 0;
    int fine_bins = 0;
    unsigned long long fine[PILOTPROXY_NUM_WEIGHT_TERMS][PILOTPROXY_FINE_NUM_BINS];
    unsigned long long powers[PILOTPROXY_NUM_WEIGHT_TERMS];

    (void)sizeof(InputType);
    PilotProxy_GetSpecs(
        &detector_window_samples,
        &num_weight_terms,
        &sample_bits_per_component,
        &reference_offset_bins);
    PilotProxy_GetVersion(0, 0, 0);
    PilotProxy_GetFineSpecs(&windows_per_stream, &pad_factor, &fine_bins);
    (void)PilotProxy_Create(0, 1);
    (void)PilotProxy_Create_Batch(0, 1, 1);
    (void)PilotProxy_Supports_FusedFine();
    PilotProxy_Compute_FusedFine_U64(0, 0, 0, 0, 0);
    (void)PilotProxy_Supports_FusedFineMaskWithSupport();
    PilotProxy_Compute_FusedFineMaskWithSupport_U64(
        0, 0, 0, 0, 0, 0, 1ULL, 0, 0, 0, 0, 0, 0);
    pilotproxy_ref_fine_powers_u64(0, 0, fine);
    pilotproxy_ref_powers_from_fine_u64(fine, powers);
}
