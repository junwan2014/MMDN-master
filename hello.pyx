import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=2] _get_subpixel_from_kpts(np.ndarray[np.float32_t, ndim=3] kpts, np.ndarray[np.float32_t, ndim=4] heatmaps, int cut_szie):
    cdef np.ndarray[np.float32_t, ndim=2] hm_part, tmp_a, tmp_b, qq, rr, ss, cc, rr1
    cdef np.ndarray[np.float32_t, ndim=2] kpt_tmp
    cdef np.ndarray[np.float32_t, ndim=1] kpt
    cdef np.ndarray[np.float32_t, ndim=2] hm
    cdef int kpt_x, kpt_y, left_x, right_x, bottom_y, top_y, m_iN,iPos
    cdef int  num_lmk = heatmaps.shape[1]
    cdef h = heatmaps.shape[2]
    kpt_tmp = np.zeros((num_lmk,2), dtype=np.float32)
    for land_index in range(num_lmk):
        hm = heatmaps[0, land_index, :, :] #取第一个特征点的热力图
        kpt = kpts[0,land_index,:] #取第一个特征点的坐标
        kpt_x = int(kpt[0])
        kpt_y = int(kpt[1])
        if kpt_x-cut_szie<0:
            left_x = 0
            right_x = 2*cut_szie+1
        elif kpt_x+cut_szie>127:
            left_x = 127-(2*cut_szie+1)
            right_x = 127
        else:
            left_x = kpt_x-cut_szie
            right_x = kpt_x+cut_szie

        if kpt_y-cut_szie<0:
            bottom_y = 0
            top_y = 2*cut_szie+1
        elif kpt_y+cut_szie>127:
            bottom_y = 127-(2*cut_szie+1)
            top_y = 127
        else:
            bottom_y = kpt_y-cut_szie
            top_y = kpt_y+cut_szie

        hm_part = hm[bottom_y:top_y+1, left_x:right_x+1]
        m_iN = (2*cut_szie+1) * (2*cut_szie+1)
        iPos = 0
        tmp_a = np.zeros((m_iN,1), dtype=np.float32)
        tmp_b = np.zeros((m_iN,5), dtype=np.float32)
        for ii in range(2*cut_szie+1):
            for jj in range(2*cut_szie+1):
                tmp = hm_part[ii,jj]
                tmp_a[iPos,0] = tmp*np.log(float(tmp))
                tmp_b[iPos,0] = tmp
                tmp_b[iPos, 1] = tmp*ii
                tmp_b[iPos, 2] = tmp*jj
                tmp_b[iPos, 3] = tmp*ii*ii
                tmp_b[iPos, 4] = tmp*jj*jj
                iPos = iPos + 1
        qq, rr = np.linalg.qr(tmp_b,mode='complete')
        ss =np.dot(qq.T,tmp_a)
        ss = ss[0:5]
        rr1 = rr[0:5, 0:5]
        cc = np.dot(np.linalg.inv(rr1),ss)
        kpt_tmp[land_index,0] = left_x - 0.5 * cc[2] / cc[4]
        kpt_tmp[land_index, 1] = bottom_y - 0.5 * cc[1] / cc[3]
    return kpt_tmp

def get_subpixel_from_kpts(a, b,c):
    return _get_subpixel_from_kpts(a, b, c)