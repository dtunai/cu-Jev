#ifndef CUJEV_SAFETENSORS_H
#define CUJEV_SAFETENSORS_H

#include "cujev/cujev.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { ST_BF16 = 0, ST_F16 = 1, ST_F32 = 2, ST_OTHER = 3 } st_dtype;

typedef struct st_tensor {
    const char *name;
    st_dtype dtype;
    int ndim;
    int64_t shape[8];
    const void *data; /* points into the mmap */
    size_t nbytes;
} st_tensor;

typedef struct st_file st_file;

/* A set of shards read from checkpoint_dir/model.safetensors.index.json (or a
 * single model.safetensors). Files are mmap'd; nothing is copied. */
typedef struct st_bundle {
    st_file **files;
    int num_files;
    st_tensor *tensors;
    int num_tensors;
} st_bundle;

cujev_status st_bundle_open(const char *checkpoint_dir, st_bundle *out);
void st_bundle_close(st_bundle *b);
const st_tensor *st_bundle_find(const st_bundle *b, const char *name);
size_t st_numel(const st_tensor *t);

#ifdef __cplusplus
}
#endif
#endif
