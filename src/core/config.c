#include "config.h"
#include "cJSON.h"
#include "util.h"

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[n] = 0;
    fclose(f);
    if (len)
        *len = (size_t)n;
    return buf;
}

static int geti(const cJSON *o, const char *k, int dflt) {
    const cJSON *v = cJSON_GetObjectItemCaseSensitive(o, k);
    return cJSON_IsNumber(v) ? (int)v->valuedouble : dflt;
}
static double getd(const cJSON *o, const char *k, double dflt) {
    const cJSON *v = cJSON_GetObjectItemCaseSensitive(o, k);
    return cJSON_IsNumber(v) ? v->valuedouble : dflt;
}
static int getb(const cJSON *o, const char *k, int dflt) {
    const cJSON *v = cJSON_GetObjectItemCaseSensitive(o, k);
    return cJSON_IsBool(v) ? cJSON_IsTrue(v) : dflt;
}

cujev_status cujev_config_load(const char *dir, cujev_config *c) {
    char path[4096];
    snprintf(path, sizeof path, "%s/config.json", dir);
    char *text = read_file(path, NULL);
    if (!text)
        return cujev_fail(CUJEV_ERR_IO, "cannot read %s", path);
    cJSON *root = cJSON_Parse(text);
    free(text);
    if (!root)
        return cujev_fail(CUJEV_ERR_FORMAT, "%s: invalid JSON", path);

    const cJSON *t = cJSON_GetObjectItemCaseSensitive(root, "text_config");
    if (!cJSON_IsObject(t))
        t = root; /* text-only checkpoints keep everything at the top level */

    memset(c, 0, sizeof *c);
    const cJSON *mt = cJSON_GetObjectItemCaseSensitive(t, "model_type");
    if (!cJSON_IsString(mt) || strncmp(mt->valuestring, "qwen3_5", 7) != 0) {
        cJSON_Delete(root);
        return cujev_fail(CUJEV_ERR_UNSUPPORTED, "model_type %s is not qwen3_5",
                          cJSON_IsString(mt) ? mt->valuestring : "(missing)");
    }
    c->hidden_size = geti(t, "hidden_size", 0);
    c->intermediate_size = geti(t, "intermediate_size", 0);
    c->num_layers = geti(t, "num_hidden_layers", 0);
    c->vocab_size = geti(t, "vocab_size", 0);
    c->rms_norm_eps = (float)getd(t, "rms_norm_eps", 1e-6);
    c->tie_word_embeddings = getb(t, "tie_word_embeddings", getb(root, "tie_word_embeddings", 0));
    c->num_heads = geti(t, "num_attention_heads", 0);
    c->num_kv_heads = geti(t, "num_key_value_heads", 0);
    c->head_dim = geti(t, "head_dim", c->hidden_size / (c->num_heads ? c->num_heads : 1));
    c->attn_output_gate = getb(t, "attn_output_gate", 0);
    c->lin_num_k_heads = geti(t, "linear_num_key_heads", 0);
    c->lin_num_v_heads = geti(t, "linear_num_value_heads", 0);
    c->lin_k_dim = geti(t, "linear_key_head_dim", 0);
    c->lin_v_dim = geti(t, "linear_value_head_dim", 0);
    c->lin_conv_kernel = geti(t, "linear_conv_kernel_dim", 4);

    const cJSON *rp = cJSON_GetObjectItemCaseSensitive(t, "rope_parameters");
    double partial = 1.0;
    c->rope_theta = 10000.0;
    if (cJSON_IsObject(rp)) {
        c->rope_theta = getd(rp, "rope_theta", 10000.0);
        partial = getd(rp, "partial_rotary_factor", 1.0);
    } else {
        c->rope_theta = getd(t, "rope_theta", 10000.0);
        partial = getd(t, "partial_rotary_factor", 1.0);
    }
    c->rotary_dim = (int)(c->head_dim * partial);

    const cJSON *lt = cJSON_GetObjectItemCaseSensitive(t, "layer_types");
    if (!cJSON_IsArray(lt) || cJSON_GetArraySize(lt) != c->num_layers ||
        c->num_layers > CUJEV_MAX_LAYERS) {
        cJSON_Delete(root);
        return cujev_fail(CUJEV_ERR_FORMAT, "layer_types missing or wrong length");
    }
    int i = 0;
    const cJSON *e;
    cJSON_ArrayForEach(e, lt) {
        if (!cJSON_IsString(e)) {
            cJSON_Delete(root);
            return cujev_fail(CUJEV_ERR_FORMAT, "layer_types[%d] is not a string", i);
        }
        if (strcmp(e->valuestring, "full_attention") == 0) {
            c->layer_types[i] = CUJEV_LAYER_FULL;
            c->num_full_layers++;
        } else if (strcmp(e->valuestring, "linear_attention") == 0) {
            c->layer_types[i] = CUJEV_LAYER_LINEAR;
            c->num_linear_layers++;
        } else {
            cJSON_Delete(root);
            return cujev_fail(CUJEV_ERR_UNSUPPORTED, "layer type %s", e->valuestring);
        }
        i++;
    }
    cJSON_Delete(root);

    if (c->hidden_size <= 0 || c->num_layers <= 0 || c->vocab_size <= 0 || c->num_heads <= 0 ||
        c->lin_num_v_heads <= 0)
        return cujev_fail(CUJEV_ERR_FORMAT, "config.json: missing required fields");
    if (c->lin_num_v_heads % c->lin_num_k_heads != 0 || c->num_heads % c->num_kv_heads != 0)
        return cujev_fail(CUJEV_ERR_UNSUPPORTED, "head counts must divide");
    if (c->lin_k_dim != 128 || c->lin_v_dim != 128 || c->head_dim != 256 || c->rotary_dim != 64)
        return cujev_fail(CUJEV_ERR_UNSUPPORTED,
                          "kernels are specialised for k=v=128, head_dim=256, rotary=64 "
                          "(got %d/%d/%d/%d)",
                          c->lin_k_dim, c->lin_v_dim, c->head_dim, c->rotary_dim);
    if (c->num_heads / c->num_kv_heads != 4)
        return cujev_fail(CUJEV_ERR_UNSUPPORTED, "attention kernel needs 4 q-heads per kv-head (got %d)",
                          c->num_heads / c->num_kv_heads);
    if (c->lin_conv_kernel != 4)
        return cujev_fail(CUJEV_ERR_UNSUPPORTED, "conv kernel %d (need 4)", c->lin_conv_kernel);

    /* A human-readable name derived from the width (the checkpoint does not say). */
    const char *size = c->hidden_size == 1024 ? "0.8B" : c->hidden_size == 2048 ? "2B"
                     : c->hidden_size == 2560 ? "4B" : c->hidden_size == 4096 ? "9B"
                     : c->hidden_size == 5120 ? "27B" : NULL;
    if (size)
        snprintf(c->name, sizeof c->name, "Qwen3.5-%s", size);
    else
        snprintf(c->name, sizeof c->name, "Qwen3.5-h%d-L%d", c->hidden_size, c->num_layers);
    return CUJEV_OK;
}
