.text
.globl native_attention_scores_probe
native_attention_scores_probe:
    s_waitcnt lgkmcnt(0)
    s_barrier
    s_add_u32 s0, s65, s8
    s_mul_i32 s0, s0, 6144
    v_lshlrev_b32 v3, 1, v0
    v_add_u32 v3, s0, v3
    v_lshlrev_b32 v1, 1, v0
    v_add_u32 v1, 128, v1
    .rept 12
    ds_read_u16 v2, v1
    s_waitcnt lgkmcnt(0)
    global_store_short v3, v2, s[42:43]
    v_add_u32 v1, 512, v1
    v_add_u32 v3, 512, v3
    .endr
    s_endpgm
