// Entry point for the classic physics lane, one work item per race. The
// scratch buffers are private to one race, so no synchronization is needed.

__kernel void race_classic(__global const unsigned int* ri_all,
    __global const double* rd_all, __global unsigned int* su_all,
    __global double* sd_all, __global double* frames_all,
    __global unsigned int* st_all, __global unsigned int* rp_all,
    const unsigned int slice_end, const unsigned int resume) {
    const int id = get_global_id(0);
    __global const unsigned int* ri = ri_all + (long)id * CLS_RI_STRIDE;
    const int n = (int)ri[CLS_RI_N];
    if (n <= 0) {
        return;
    }
    __global const double* rd = rd_all + (long)id * CLS_RD_STRIDE;
    __global unsigned int* su = su_all + (long)id * CLS_SU_STRIDE;
    __global double* sd = sd_all + (long)id * CLS_SD_STRIDE;
    __global unsigned int* st = st_all + (long)id * CLS_ST_STRIDE;
    if (resume != 0u) {
        // A lane the host already considers finished or flagged stays put.
        if (st[S_DONE] != 0u || st[S_FALLBACK] != 0u) {
            return;
        }
    } else {
        st[S_FALLBACK] = 0u;
        st[S_TICKS] = 0u;
        st[S_RPOVER] = 0u;
        st[S_DONE] = 0u;
    }
    const int nmax = (int)ri[CLS_RI_NMAX];
    __global double* fr =
        frames_all + (long)id * CLS_MAX_TICKS * (long)nmax * CLS_FRAME_COLS;
    Cc c;
    c.su = su;
    c.sd = sd;
    c.cd = rd;
    c.rp = rp_all + (long)id * CLS_RP_STRIDE;
    c.rp_over = 0;
    c.n = n;
    c.nhills = (int)ri[CLS_RI_NSLOPES];
    if (n > CLS_MAXR || nmax > CLS_MAXR || c.nhills > CLS_MAX_SLOPES) {
        st[S_FALLBACK] = 3u;
    } else {
        cc_run(&c, ri, st, fr, slice_end, resume);
        if (su[U_CAP]) {
            st[S_FALLBACK] = 1u;
        }
    }
}
