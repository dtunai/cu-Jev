#include "safetensors.h"
#include "cJSON.h"
#include "util.h"
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct st_file {
    char *path;
    int fd;
    void *map;
    size_t size;
    cJSON *header; /* keeps tensor name strings alive */
};

static int cmp_tensor(const void *a, const void *b) {
    return strcmp(((const st_tensor *)a)->name, ((const st_tensor *)b)->name);
}

static cujev_status st_file_open(const char *path, st_file **out, st_tensor **tensors, int *count,
                                 int *cap) {
    st_file *f = (st_file *)calloc(1, sizeof *f);
    f->path = strdup(path);
    f->fd = open(path, O_RDONLY);
    if (f->fd < 0)
        return cujev_fail(CUJEV_ERR_IO, "open %s failed", path);
    struct stat st;
    fstat(f->fd, &st);
    f->size = (size_t)st.st_size;
    f->map = mmap(NULL, f->size, PROT_READ, MAP_PRIVATE, f->fd, 0);
    if (f->map == MAP_FAILED)
        return cujev_fail(CUJEV_ERR_IO, "mmap %s failed", path);
    if (f->size < 8)
        return cujev_fail(CUJEV_ERR_FORMAT, "%s: too small", path);
    uint64_t hlen;
    memcpy(&hlen, f->map, 8);
    if (hlen + 8 > f->size)
        return cujev_fail(CUJEV_ERR_FORMAT, "%s: bad header length", path);
    char *hjson = (char *)malloc(hlen + 1);
    memcpy(hjson, (char *)f->map + 8, hlen);
    hjson[hlen] = 0;
    f->header = cJSON_Parse(hjson);
    free(hjson);
    if (!f->header)
        return cujev_fail(CUJEV_ERR_FORMAT, "%s: header is not JSON", path);
    const char *base = (char *)f->map + 8 + hlen;
    size_t data_size = f->size - 8 - hlen;

    const cJSON *e;
    cJSON_ArrayForEach(e, f->header) {
        if (strcmp(e->string, "__metadata__") == 0)
            continue;
        const cJSON *dt = cJSON_GetObjectItemCaseSensitive(e, "dtype");
        const cJSON *sh = cJSON_GetObjectItemCaseSensitive(e, "shape");
        const cJSON *off = cJSON_GetObjectItemCaseSensitive(e, "data_offsets");
        if (!cJSON_IsString(dt) || !cJSON_IsArray(sh) || !cJSON_IsArray(off) ||
            cJSON_GetArraySize(off) != 2)
            return cujev_fail(CUJEV_ERR_FORMAT, "%s: malformed entry %s", path, e->string);
        if (*count == *cap) {
            *cap = *cap ? *cap * 2 : 256;
            *tensors = (st_tensor *)realloc(*tensors, (size_t)*cap * sizeof(st_tensor));
        }
        st_tensor *t = &(*tensors)[(*count)++];
        memset(t, 0, sizeof *t);
        t->name = e->string;
        if (strcmp(dt->valuestring, "BF16") == 0)
            t->dtype = ST_BF16;
        else if (strcmp(dt->valuestring, "F16") == 0)
            t->dtype = ST_F16;
        else if (strcmp(dt->valuestring, "F32") == 0)
            t->dtype = ST_F32;
        else
            t->dtype = ST_OTHER;
        t->ndim = cJSON_GetArraySize(sh);
        if (t->ndim > 8)
            return cujev_fail(CUJEV_ERR_FORMAT, "%s: tensor %s has %d dims", path, t->name, t->ndim);
        for (int i = 0; i < t->ndim; i++)
            t->shape[i] = (int64_t)cJSON_GetArrayItem(sh, i)->valuedouble;
        size_t a = (size_t)cJSON_GetArrayItem(off, 0)->valuedouble;
        size_t b = (size_t)cJSON_GetArrayItem(off, 1)->valuedouble;
        if (b < a || b > data_size)
            return cujev_fail(CUJEV_ERR_FORMAT, "%s: tensor %s offsets out of range", path, t->name);
        t->data = base + a;
        t->nbytes = b - a;
    }
    *out = f;
    return CUJEV_OK;
}

static void st_file_close(st_file *f) {
    if (!f)
        return;
    if (f->header)
        cJSON_Delete(f->header);
    if (f->map && f->map != MAP_FAILED)
        munmap(f->map, f->size);
    if (f->fd >= 0)
        close(f->fd);
    free(f->path);
    free(f);
}

cujev_status st_bundle_open(const char *dir, st_bundle *b) {
    memset(b, 0, sizeof *b);
    int cap = 0;
    /* Collect shard names: every *.safetensors in the directory. */
    DIR *d = opendir(dir);
    if (!d)
        return cujev_fail(CUJEV_ERR_IO, "cannot open directory %s", dir);
    char **names = NULL;
    int n = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        size_t L = strlen(de->d_name);
        if (L > 12 && strcmp(de->d_name + L - 12, ".safetensors") == 0) {
            names = (char **)realloc(names, (size_t)(n + 1) * sizeof(char *));
            names[n++] = strdup(de->d_name);
        }
    }
    closedir(d);
    if (n == 0)
        return cujev_fail(CUJEV_ERR_IO, "no *.safetensors in %s", dir);
    b->files = (st_file **)calloc((size_t)n, sizeof(st_file *));
    for (int i = 0; i < n; i++) {
        char path[4096];
        snprintf(path, sizeof path, "%s/%s", dir, names[i]);
        free(names[i]);
        cujev_status s = st_file_open(path, &b->files[i], &b->tensors, &b->num_tensors, &cap);
        if (s != CUJEV_OK) {
            free(names);
            st_bundle_close(b);
            return s;
        }
        b->num_files++;
    }
    free(names);
    qsort(b->tensors, (size_t)b->num_tensors, sizeof(st_tensor), cmp_tensor);
    return CUJEV_OK;
}

void st_bundle_close(st_bundle *b) {
    for (int i = 0; i < b->num_files; i++)
        st_file_close(b->files[i]);
    free(b->files);
    free(b->tensors);
    memset(b, 0, sizeof *b);
}

const st_tensor *st_bundle_find(const st_bundle *b, const char *name) {
    st_tensor key;
    key.name = name;
    return (const st_tensor *)bsearch(&key, b->tensors, (size_t)b->num_tensors, sizeof(st_tensor),
                                      cmp_tensor);
}

size_t st_numel(const st_tensor *t) {
    size_t n = 1;
    for (int i = 0; i < t->ndim; i++)
        n *= (size_t)t->shape[i];
    return n;
}
