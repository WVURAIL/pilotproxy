# coding=utf-8
"""Stage names of the synthetic fine-detector benchmark."""

STAGE_IDEAL_TONE_FLOAT = "ideal_tone_float"
STAGE_ATSC_FLOAT = "atsc_8vsb_float"
STAGE_INPUT_INT4 = "atsc_8vsb_input_int4_float"
STAGE_WEIGHT_INT4 = "atsc_8vsb_weight_int4_float"
STAGE_JOINT_INT4_FLOAT = "atsc_8vsb_joint_int4_float_transform"
STAGE_FIXED_FLOAT_DECISION = "atsc_8vsb_fixed_transform_float_decision"
STAGE_FIXED_Q16_CPU = "atsc_8vsb_fixed_transform_q16_cpu"
STAGE_FULL_GPU = "atsc_8vsb_packed_gpu_fine_exact_q16"

FLOAT_RESPONSE_STAGES = (
    STAGE_IDEAL_TONE_FLOAT,
    STAGE_ATSC_FLOAT,
    STAGE_INPUT_INT4,
    STAGE_WEIGHT_INT4,
    STAGE_JOINT_INT4_FLOAT,
    STAGE_FIXED_FLOAT_DECISION,
)
