// DSO-AI Runtime (v1) — Qwen2 inference engine
// Layer-by-layer streaming from disk, static memory arenas, OpenMP GEMM.
// Compile: g++ -O3 -fopenmp -std=c++17 dso_runtime.cpp -o dso_runtime
//
// Design (per DSO manifest modules 2-3, made real):
//   * Weights live in the safetensors file, mmap'd read-only.
//   * Exactly ONE weight matrix is resident in RAM at a time (Active Weight
//     Arena, weight_buf). Each projection is streamed from disk, converted
//     BF16->FP32 on the fly, used, then overwritten. Peak RSS is independent
//     of model size: one layer's biggest weight + activations + KV-cache.
//   * Embedding / lm_head are tiled: logits are computed by streaming vocab
//     rows from disk in blocks (tiling instead of holding the whole matrix).
//   * KV-cache is a single static buffer (ring semantics: position index mod
//     MAX_SEQ).
//   * ASYNC STREAMING (manifest module 3): a background thread issues
//     MADV_WILLNEED on the NEXT layer while the current layer is computed, and
//     MADV_DONTNEED evicts a layer's pages from the OS page-cache right after
//     use. So the page-cache holds only a sliding window of ~1-2 layers + embed,
//     never the whole model. Reads are cheap (no SSD wear; only writes wear).
//   * No heap allocation inside the token generation loop.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <string>
#include <unordered_map>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>

// ---- model constants (Qwen2.5-0.5B-Instruct) ----
static const int HIDDEN    = 896;
static const int INTER     = 4864;
static const int N_LAYERS  = 24;
static const int N_HEADS   = 14;
static const int N_KV      = 2;
static const int HEAD_DIM  = HIDDEN / N_HEADS; // 64
static const int VOCAB     = 151936;
static const float EPS     = 1e-6f;
static const float ROPE_THETA = 1000000.0f;
static const int MAX_SEQ   = 2048;
static const int N_GROUPS  = N_HEADS / N_KV;    // 7

// ---- bf16 -> f32 ----
static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f; std::memcpy(&f, &u, 4); return f;
}

// ---- mmap'd model ----
static int   g_fd = -1;
static char* g_map = nullptr;
static size_t g_map_size = 0;
static uint64_t g_data_start = 0; // = 8 + header_len, absolute file offset of tensor data

struct TInfo { size_t off; std::vector<int> shape; };
static std::unordered_map<std::string, TInfo> g_tmap;
static const uint16_t* g_emb = nullptr; // embed_tokens BF16 rows [VOCAB, HIDDEN]

// ---- int8 (DSO) model mode ----
// WT: a tensor in the .dso file. kind 0 = fp32 blob, 1 = int8 blob + per-row fp32 scale.
struct WT { int kind; const void* data; const float* scale; int N, K; };
static std::unordered_map<std::string, WT> g_w;
static int g_mode = 0; // 0 = BF16 safetensors, 1 = INT8 .dso, 2 = GGUF
static const int8_t* g_emb_i8 = nullptr;     // embed rows (int8) [VOCAB, HIDDEN]
static const float* g_emb_scale = nullptr;   // per-row scale [VOCAB]

// ---- GGUF mode ----
enum { GGML_F32 = 0, GGML_F16 = 1, GGML_Q8_0 = 8, GGML_BF16 = 30 };
struct GGUFTensor { uint64_t offset; int type; std::vector<uint64_t> ne; };
static std::unordered_map<std::string, GGUFTensor> g_gguf_tmap;
static uint64_t g_gguf_data_start = 0;
static int g_gguf_alignment = 32;
// GGUF embedding info (generic: handles any type)
static const void* g_emb_gguf = nullptr;  // raw embedding tensor data
static int g_emb_gguf_type = GGML_BF16;   // GGML type
static size_t g_emb_gguf_row_bytes = 0;   // bytes per embedding row

static inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15)) << 31;
    uint32_t exp = ((h >> 10) & 0x1f);
    uint32_t mant = (h & 0x3ff);
    uint32_t u;
    if (exp == 0) {
        u = sign | ((127 - 15) << 23) | (mant << 13);
    } else if (exp == 31) {
        u = sign | (0xff << 23) | (mant << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &u, 4); return f;
}

// active weight arena (BF16 mode): one FP32 weight matrix at a time (max = down_proj)
// In INT8 mode weights are read directly as int8 from the mmap (1 byte/param).
static int8_t g_qx[4864]; // per-row INT8 activation buffer for the SIMD GEMM

// ---- static arenas (allocated ONCE at init, never inside the loop) ----
static float weight_buf[4864 * 896];          // one weight matrix at a time (max single = down_proj)
static float bias_buf[4864];                   // small bias scratch
static float x_buf[HIDDEN];                    // residual stream for current token
static float xn_buf[HIDDEN];                   // normalized
static float attn_out[HIDDEN];
static float ffn_out[HIDDEN];
static float q_buf[N_HEADS * HEAD_DIM];
static float k_buf[N_KV * HEAD_DIM];
static float v_buf[N_KV * HEAD_DIM];
static float scores[MAX_SEQ];
static float ctx_buf[N_HEADS * HEAD_DIM];
static float gate_buf[INTER];
static float up_buf[INTER];
static float act_buf[INTER];
static float logits[VOCAB];
static int g_spec_logits[VOCAB]; // scratch for draft verification at each block position

// ---- block-forward buffers (self-speculative verification, M_MAX = 16) ----
static const int M_MAX = 16;
static float x_block[M_MAX * HIDDEN];
static float xn_block_b[M_MAX * HIDDEN];
static float q_block_b[M_MAX * HIDDEN];
static float k_block_b[M_MAX * N_KV * HEAD_DIM];
static float v_block_b[M_MAX * N_KV * HEAD_DIM];
static float gate_block_b[M_MAX * INTER];
static float up_block_b[M_MAX * INTER];
static float act_block_b[M_MAX * INTER];
static float ffn_block_b[M_MAX * HIDDEN];

// KV-cache: per layer, [MAX_SEQ][N_KV][HEAD_DIM]
static float* g_kcache; // [N_LAYERS*MAX_SEQ*N_KV*HEAD_DIM]
static float* g_vcache;

static float g_inv_freq[HEAD_DIM / 2];

// ---- parse safetensors header ----
static void parse_header() {
    uint64_t hlen;
    std::memcpy(&hlen, g_map, 8);
    g_data_start = 8 + hlen;
    std::string hdr(g_map + 8, hlen);

    size_t p = hdr.find('{') + 1;
    while (p < hdr.size()) {
        while (p < hdr.size() && (hdr[p]==' '||hdr[p]=='\n'||hdr[p]=='\r'||hdr[p]=='\t'||hdr[p]==',')) p++;
        if (p >= hdr.size() || hdr[p] == '}') break;
        if (hdr[p] != '"') { p++; continue; }
        p++;
        std::string key;
        while (p < hdr.size() && hdr[p] != '"') { key += hdr[p]; p++; }
        p++; // closing quote
        while (p < hdr.size() && hdr[p] != '{') p++;
        size_t obj_start = p;
        int depth = 0; size_t q = obj_start;
        for (; q < hdr.size(); q++) {
            if (hdr[q] == '{') depth++;
            else if (hdr[q] == '}') { depth--; if (depth == 0) break; }
        }
        std::string obj = hdr.substr(obj_start + 1, q - obj_start - 1);

        TInfo info; info.off = 0;
        size_t dpos = obj.find("\"data_offsets\"");
        if (dpos != std::string::npos) {
            size_t b1 = obj.find('[', dpos);
            size_t b2 = obj.find(',', b1);
            info.off = (size_t)strtoull(obj.substr(b1 + 1, b2 - b1 - 1).c_str(), nullptr, 10);
        }
        size_t spos = obj.find("\"shape\"");
        if (spos != std::string::npos) {
            size_t s1 = obj.find('[', spos);
            size_t s2 = obj.find(']', s1);
            std::string body = obj.substr(s1 + 1, s2 - s1 - 1);
            size_t i = 0;
            while (i < body.size()) {
                while (i < body.size() && (body[i] < '0' || body[i] > '9') && body[i] != '-') i++;
                if (i >= body.size()) break;
                size_t j = i;
                while (j < body.size() && ((body[j] >= '0' && body[j] <= '9') || body[j] == '-')) j++;
                info.shape.push_back(std::atoi(body.substr(i, j - i).c_str()));
                i = j;
            }
        }
        g_tmap[key] = info;
        p = q + 1;
    }
    auto it = g_tmap.find("model.embed_tokens.weight");
    if (it == g_tmap.end()) { fprintf(stderr, "embed_tokens not found\n"); exit(1); }
    g_emb = (const uint16_t*)(g_map + g_data_start + it->second.off);
}

// ---- GGUF header parser ----
static uint64_t gguf_read_u64(const char*& p) { uint64_t v; memcpy(&v, p, 8); p += 8; return v; }
static uint32_t gguf_read_u32(const char*& p) { uint32_t v; memcpy(&v, p, 4); p += 4; return v; }
static std::string gguf_read_str(const char*& p) { uint64_t len = gguf_read_u64(p); std::string s(p, len); p += len; return s; }
static inline int gguf_elem_size(int type);
static inline size_t gguf_row_nbytes(int type, size_t row_elems);
static inline size_t gguf_tensor_bytes(const GGUFTensor& t);
static std::string gguf_resolve_name(const std::string& hf);
static void parse_gguf() {
    const char* p = g_map;
    uint32_t magic = gguf_read_u32(p);
    if (magic != 0x46554747) { fprintf(stderr, "bad GGUF magic 0x%x\n", magic); exit(1); }
    int version = (int)gguf_read_u32(p);
    uint64_t tensor_cnt = gguf_read_u64(p);
    uint64_t meta_cnt = gguf_read_u64(p);
    g_gguf_alignment = 32;
    for (uint64_t i = 0; i < meta_cnt; i++) {
        std::string key = gguf_read_str(p);
        uint32_t vtype = gguf_read_u32(p);
        auto skip_str = [&]() { uint64_t sl = gguf_read_u64(p); p += sl; };
        auto skip = [&](int n) { p += n; };
        bool is_align = (key == "general.alignment");
        switch (vtype) {
            case 0: case 1: case 2: case 3: case 8:
                if (is_align) { g_gguf_alignment = (int)gguf_read_u64(p); } else skip(8);
                break;
            case 4: skip(4); break;
            case 5: skip(1); break;
            case 6: skip_str(); break;
            case 7: {
                gguf_read_u32(p); // arr_type
                uint64_t arr_len = gguf_read_u64(p);
                for (uint64_t j = 0; j < arr_len; j++) skip_str();
                break;
            }
            case 9: skip(8); break;
            case 10: case 11: skip(2); break;
            default: fprintf(stderr, "unknown GGUF metadata type %u\n", vtype); exit(1);
        }
    }
    for (uint64_t i = 0; i < tensor_cnt; i++) {
        std::string name = gguf_read_str(p);
        uint32_t n_dims = gguf_read_u32(p);
        std::vector<uint64_t> ne(n_dims);
        for (uint32_t d = 0; d < n_dims; d++) ne[d] = gguf_read_u64(p);
        uint32_t type = gguf_read_u32(p);
        uint64_t offset = gguf_read_u64(p);
        g_gguf_tmap[name] = {offset, (int)type, ne};
        // GGUF v3+: each tensor-info entry is padded so the file position stays
        // 32-byte aligned (GGUF_TYPE_SIZE).
        if (version >= 3) {
            size_t pos = (size_t)(p - g_map);
            size_t pad = (32 - (pos % 32)) % 32;
            p += pad;
        }
    }
    size_t total_hdr = (size_t)(p - g_map);
    size_t ds = (total_hdr + g_gguf_alignment - 1) & ~(size_t)(g_gguf_alignment - 1);
    g_gguf_data_start = ds;
    // set up embedding pointer (try HF then llama.cpp naming)
    std::string emb_name = gguf_resolve_name("model.embed_tokens.weight");
    if (emb_name.empty()) { fprintf(stderr, "embed_tokens not found\n"); exit(1); }
    auto emb = g_gguf_tmap.find(emb_name);
    g_emb_gguf = g_map + ds + emb->second.offset;
    g_emb_gguf_type = emb->second.type;
    size_t n_elems = 1; for (auto d : emb->second.ne) n_elems *= d;
    size_t n_rows = emb->second.ne.size() > 1 ? emb->second.ne[0] : 1;
    g_emb_gguf_row_bytes = gguf_row_nbytes(g_emb_gguf_type, n_elems / n_rows);
    if (g_emb_gguf_type == GGML_F32) {
        g_emb = (const uint16_t*)((const float*)g_emb_gguf); // reuse for F32 (will cast in lm_head)
    } else if (g_emb_gguf_type == GGML_BF16) {
        g_emb = (const uint16_t*)g_emb_gguf;
    }
    if (getenv("GGUF_DEBUG")) {
        for (auto& kv : g_gguf_tmap) {
            fprintf(stderr, "  [gguf] tensor '%s' type=%d shape=[", kv.first.c_str(), kv.second.type);
            for (size_t i = 0; i < kv.second.ne.size(); i++) fprintf(stderr, "%s%llu", i?"x":"", (unsigned long long)kv.second.ne[i]);
            fprintf(stderr, "]\n");
        }
    }
    fprintf(stderr, "[gguf] v=%d tensors=%llu alignment=%d embed='%s' type=%d\n",
            version, (unsigned long long)tensor_cnt, g_gguf_alignment, emb_name.c_str(), g_emb_gguf_type);
}
static inline int gguf_elem_size(int type) {
    switch (type) {
        case GGML_F32: return 4;
        case GGML_F16: case GGML_BF16: return 2;
        case GGML_Q8_0: return 1; // per-element after dequant, but 34 bytes per 32 elems
        default: return 2;
    }
}
static inline size_t gguf_row_nbytes(int type, size_t row_elems) {
    if (type == GGML_Q8_0) {
        size_t n_blocks = (row_elems + 31) / 32;
        return n_blocks * 34; // 2-byte fp16 scale + 32 int8 = 34 bytes per block
    }
    return row_elems * (size_t)gguf_elem_size(type);
}
static inline size_t gguf_tensor_bytes(const GGUFTensor& t) {
    size_t n = 1; for (auto d : t.ne) n *= (size_t)d;
    if (t.type == GGML_Q8_0) return (n / 32) * 34 + (n % 32 ? 34 : 0);
    return n * (size_t)gguf_elem_size(t.type);
}

// Resolve an HF-style tensor name to an actual GGUF tensor name.
// Handles both HF naming (model.layers.N.self_attn.q_proj.weight) and
// llama.cpp GGUF naming (blk.N.attn_q.weight), plus embed/norm/output.
static std::string gguf_resolve_name(const std::string& hf) {
    if (g_gguf_tmap.count(hf)) return hf;
    auto try_cands = [](std::initializer_list<const char*> cands) -> std::string {
        for (auto c : cands) if (g_gguf_tmap.count(c)) return c;
        return "";
    };
    if (hf == "model.embed_tokens.weight")
        return try_cands({"token_embd.weight", "model.embed_tokens.weight"});
    if (hf == "model.norm.weight")
        return try_cands({"output_norm.weight", "model.norm.weight"});
    if (hf == "lm_head.weight")
        return try_cands({"output.weight", "token_embd.weight", "lm_head.weight"});
    const char* p = "model.layers.";
    if (hf.rfind(p, 0) == 0) {
        size_t dot = hf.find('.', strlen(p));
        std::string L = hf.substr(strlen(p), dot - strlen(p));
        std::string rest = hf.substr(dot + 1);
        std::string cand;
        if (rest == "input_layernorm.weight") cand = "blk." + L + ".attn_norm.weight";
        else if (rest == "post_attention_layernorm.weight") cand = "blk." + L + ".ffn_norm.weight";
        else if (rest == "self_attn.q_proj.weight") cand = "blk." + L + ".attn_q.weight";
        else if (rest == "self_attn.k_proj.weight") cand = "blk." + L + ".attn_k.weight";
        else if (rest == "self_attn.v_proj.weight") cand = "blk." + L + ".attn_v.weight";
        else if (rest == "self_attn.o_proj.weight") cand = "blk." + L + ".attn_output.weight";
        else if (rest == "mlp.gate_proj.weight") cand = "blk." + L + ".ffn_gate.weight";
        else if (rest == "mlp.up_proj.weight") cand = "blk." + L + ".ffn_up.weight";
        else if (rest == "mlp.down_proj.weight") cand = "blk." + L + ".ffn_down.weight";
        if (!cand.empty()) {
            std::string r = try_cands({cand.c_str(), hf.c_str()});
            if (!r.empty()) return r;
        }
    }
    return "";
}

static void load_weight_gguf(const std::string& name, float* dst) {
    std::string rn = gguf_resolve_name(name);
    if (rn.empty()) { fprintf(stderr, "MISSING GGUF tensor: %s\n", name.c_str()); exit(1); }
    auto it = g_gguf_tmap.find(rn);
    const char* src = g_map + g_gguf_data_start + it->second.offset;
    size_t n = 1; for (auto d : it->second.ne) n *= (size_t)d;
    if (it->second.type == GGML_F32) {
        memcpy(dst, src, n * 4);
    } else if (it->second.type == GGML_BF16) {
        const uint16_t* b = (const uint16_t*)src;
        #if defined(__x86_64__) || defined(__i386__)
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m128i bf16 = _mm_loadu_si128((const __m128i*)&b[i]);
            __m256i i32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16), 16);
            _mm256_storeu_ps(&dst[i], _mm256_castsi256_ps(i32));
        }
        for (; i < n; i++) dst[i] = bf16_to_f32(b[i]);
        #else
        for (size_t i = 0; i < n; i++) dst[i] = bf16_to_f32(b[i]);
        #endif
    } else if (it->second.type == GGML_F16) {
        const uint16_t* b = (const uint16_t*)src;
        for (size_t i = 0; i < n; i++) dst[i] = fp16_to_f32(b[i]);
    } else if (it->second.type == GGML_Q8_0) {
        int n_blocks = (int)(n + 31) / 32;
        #pragma omp parallel for
        for (int b = 0; b < n_blocks; b++) {
            const uint8_t* blk = (const uint8_t*)src + b * 34;
            float d = fp16_to_f32(*(const uint16_t*)blk);
            const int8_t* qs = (const int8_t*)(blk + 2);
            int rem = (int)n - b * 32;
            int bn = rem > 32 ? 32 : rem;
            for (int i = 0; i < bn; i++) dst[b * 32 + i] = d * (float)qs[i];
        }
    } else {
        fprintf(stderr, "unsupported GGUF type %d for %s\n", it->second.type, name.c_str());
        exit(1);
    }
}
// returns false (and leaves dst untouched) if the bias tensor is absent
static bool load_bias_gguf(const std::string& name, float* dst, int n) {
    std::string rn = gguf_resolve_name(name);
    if (rn.empty()) return false; // no bias in this export
    auto it = g_gguf_tmap.find(rn);
    const char* src = g_map + g_gguf_data_start + it->second.offset;
    if (it->second.type == GGML_F32) {
        memcpy(dst, src, (size_t)n * 4);
    } else if (it->second.type == GGML_BF16) {
        const uint16_t* b = (const uint16_t*)src;
        for (int i = 0; i < n; i++) dst[i] = bf16_to_f32(b[i]);
    } else {
        fprintf(stderr, "unsupported bias type %d\n", it->second.type);
        exit(1);
    }
    return true;
}

static void load_weight(const std::string& name, float* dst) {
    auto it = g_tmap.find(name);
    if (it == g_tmap.end()) { fprintf(stderr, "MISSING tensor: %s\n", name.c_str()); exit(1); }
    const uint16_t* src = (const uint16_t*)(g_map + g_data_start + it->second.off);
    size_t n = 1; for (int s : it->second.shape) n *= (size_t)s;
#if defined(__x86_64__) || defined(__i386__)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i bf16 = _mm_loadu_si128((const __m128i*)&src[i]);
        __m256i i32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16), 16);
        _mm256_storeu_ps(&dst[i], _mm256_castsi256_ps(i32));
    }
    for (; i < n; i++) dst[i] = bf16_to_f32(src[i]);
#else
    for (size_t i = 0; i < n; i++) dst[i] = bf16_to_f32(src[i]);
#endif
}
static void load_bias(const std::string& name, float* dst, int n) {
    auto it = g_tmap.find(name);
    if (it == g_tmap.end()) { fprintf(stderr, "MISSING bias: %s\n", name.c_str()); exit(1); }
    const uint16_t* src = (const uint16_t*)(g_map + g_data_start + it->second.off);
#if defined(__x86_64__) || defined(__i386__)
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i bf16 = _mm_loadu_si128((const __m128i*)&src[i]);
        __m256i i32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16), 16);
        _mm256_storeu_ps(&dst[i], _mm256_castsi256_ps(i32));
    }
    for (; i < n; i++) dst[i] = bf16_to_f32(src[i]);
#else
    for (int i = 0; i < n; i++) dst[i] = bf16_to_f32(src[i]);
#endif
}

// ---- parse the INT8 .dso file (header JSON: kind/shape/off/nbytes) ----
static long parse_int_after(const std::string& s, size_t pos) {
    size_t c = s.find(':', pos);
    if (c == std::string::npos) return 0;
    size_t i = c + 1;
    while (i < s.size() && (s[i] < '0' || s[i] > '9') && s[i] != '-') i++;
    size_t j = i;
    while (j < s.size() && ((s[j] >= '0' && s[j] <= '9') || s[j] == '-')) j++;
    return std::strtol(s.substr(i, j - i).c_str(), nullptr, 10);
}
static std::vector<int> parse_shape_in(const std::string& obj) {
    std::vector<int> sh;
    size_t sp = obj.find("\"shape\"");
    if (sp == std::string::npos) return sh;
    size_t s1 = obj.find('[', sp);
    size_t s2 = obj.find(']', s1);
    std::string body = obj.substr(s1 + 1, s2 - s1 - 1);
    size_t i = 0;
    while (i < body.size()) {
        while (i < body.size() && (body[i] < '0' || body[i] > '9') && body[i] != '-') i++;
        if (i >= body.size()) break;
        size_t j = i;
        while (j < body.size() && ((body[j] >= '0' && body[j] <= '9') || body[j] == '-')) j++;
        sh.push_back(std::atoi(body.substr(i, j - i).c_str()));
        i = j;
    }
    return sh;
}
static void parse_dso() {
    uint64_t hlen; std::memcpy(&hlen, g_map, 8);
    g_data_start = 8 + hlen;
    std::string hdr(g_map + 8, hlen);
    size_t p = hdr.find('{') + 1;
    while (p < hdr.size()) {
        while (p < hdr.size() && (hdr[p]==' '||hdr[p]=='\n'||hdr[p]=='\r'||hdr[p]=='\t'||hdr[p]==',')) p++;
        if (p >= hdr.size() || hdr[p] == '}') break;
        if (hdr[p] != '"') { p++; continue; }
        p++; std::string key; while (p < hdr.size() && hdr[p] != '"') { key += hdr[p]; p++; }
        p++;
        while (p < hdr.size() && hdr[p] != '{') p++;
        size_t obj_start = p;
        int depth = 0; size_t q = obj_start;
        for (; q < hdr.size(); q++) {
            if (hdr[q] == '{') depth++;
            else if (hdr[q] == '}') { depth--; if (depth == 0) break; }
        }
        std::string obj = hdr.substr(obj_start + 1, q - obj_start - 1);
        WT w; w.kind = 0; w.scale = nullptr; w.N = 0; w.K = 1;
        size_t kp = obj.find("\"kind\"");
        if (kp != std::string::npos) {
            size_t c1 = obj.find('"', kp + 6);
            size_t c2 = obj.find('"', c1 + 1);
            std::string kind = obj.substr(c1 + 1, c2 - c1 - 1);
            w.kind = (kind == "int8") ? 1 : 0;
        }
        long off = parse_int_after(obj, obj.find("\"off\""));
        std::vector<int> sh = parse_shape_in(obj);
        w.N = sh.empty() ? 0 : sh[0];
        w.K = sh.size() > 1 ? sh[1] : 1;
        w.data = g_map + (size_t)off;   // .dso offsets are absolute file offsets
        if (w.kind == 1) w.scale = (const float*)(g_map + (size_t)off + (size_t)w.N * w.K);
        g_w[key] = w;
        if (key == "model.embed_tokens.weight") { g_emb_i8 = (const int8_t*)w.data; g_emb_scale = w.scale; }
        p = q + 1;
    }
}

// ---- INT8 GEMM: Y[i,j] = scale[j] * sum_k X[i,k] * Wq[j,k]  (Wq int8) ----
static void linear_i8(const float* X, int M, int K, const int8_t* W, const float* S, int N, float* Y, int nt) {
    if (M == 1) {
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int j = 0; j < N; j++) {
            const int8_t* wj = W + (size_t)j * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += (float)X[k] * (float)wj[k];
            Y[j] = acc * S[j];
        }
    } else {
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int i = 0; i < M; i++) {
            const float* xi = X + (size_t)i * K;
            float* yi = Y + (size_t)i * N;
            for (int j = 0; j < N; j++) {
                const int8_t* wj = W + (size_t)j * K;
                float acc = 0.0f;
                for (int k = 0; k < K; k++) acc += (float)xi[k] * (float)wj[k];
                yi[j] = acc * S[j];
            }
        }
    }
}

// ---- unified weight accessors (both modes) ----
static void linear(const float* X, int M, int K, const float* W, int N, float* Y, int nt); // fwd decl
#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2")))
#endif
static void linear_i8_avx2(const float* X, int M, int K, const int8_t* W, const float* S, int N, float* Y, int nt); // fwd decl
static const float* norm_weight(const std::string& name) {
    if (g_mode == 0) { load_weight(name, weight_buf); return weight_buf; }
    if (g_mode == 2) { load_weight_gguf(name, weight_buf); return weight_buf; }
    auto it = g_w.find(name);
    if (it == g_w.end()) { fprintf(stderr, "MISSING %s\n", name.c_str()); exit(1); }
    return (const float*)it->second.data;
}
static void proj(const std::string& wname, const float* X, int M, int K, int N, float* Y, int nt,
                 const std::string& bname = "") {
    if (g_mode == 0) {
        load_weight(wname, weight_buf);
        linear(X, M, K, weight_buf, N, Y, nt);
        if (!bname.empty()) {
            auto sh = g_tmap[bname].shape; int n = 1; for (int s : sh) n *= s;
            load_bias(bname, bias_buf, n);
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    Y[(size_t)i * N + j] += bias_buf[j];
        }
    } else if (g_mode == 2) {
        load_weight_gguf(wname, weight_buf);
        linear(X, M, K, weight_buf, N, Y, nt);
        if (!bname.empty()) {
            int n = N;
            if (load_bias_gguf(bname, bias_buf, n)) {
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++)
                        Y[(size_t)i * N + j] += bias_buf[j];
            }
        }
    } else {
        WT& w = g_w[wname];
#if defined(__x86_64__) || defined(__i386__)
        linear_i8_avx2(X, M, K, (const int8_t*)w.data, w.scale, N, Y, nt);
#else
        linear_i8(X, M, K, (const int8_t*)w.data, w.scale, N, Y, nt);
#endif
        if (!bname.empty()) {
            WT& b = g_w[bname];
            const float* bp = (const float*)b.data;
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    Y[(size_t)i * N + j] += bp[j];
        }
    }
}

// ---- async streaming (manifest module 3): madvise-driven page-cache window ----
static void madvise_tensor(const std::string& name, int advice) {
    if (g_mode == 0) {
        auto it = g_tmap.find(name);
        if (it == g_tmap.end()) return;
        size_t nbytes = 2; // BF16
        for (int s : it->second.shape) nbytes *= (size_t)s;
        madvise(g_map + g_data_start + it->second.off, nbytes, advice);
    } else if (g_mode == 2) {
        std::string rn = gguf_resolve_name(name);
        if (rn.empty()) return;
        auto it = g_gguf_tmap.find(rn);
        size_t nbytes = gguf_tensor_bytes(it->second);
        madvise((void*)(g_map + g_gguf_data_start + it->second.offset), nbytes, advice);
    } else {
        auto it = g_w.find(name);
        if (it == g_w.end()) return;
        const WT& w = it->second;
        size_t nbytes = (size_t)w.N * w.K;
        if (w.kind == 1) nbytes += (size_t)w.N * sizeof(float); // + scales
        madvise(const_cast<void*>(w.data), nbytes, advice);
    }
}
static void layer_madvise(int L, int advice) {
    const char* parts[] = {
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.q_proj.bias",
        "self_attn.k_proj.weight", "self_attn.k_proj.bias",
        "self_attn.v_proj.weight", "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"
    };
    for (auto p : parts) {
        std::string nm = "model.layers." + std::to_string(L) + "." + p;
        madvise_tensor(nm, advice);
    }
}
static std::mutex g_sm;
static std::condition_variable g_cv;
static int g_pending = -1;     // layer index the worker should WILLNEED (-1 = idle)
static bool g_stop = false;
static std::thread g_worker;
static void stream_worker() {
    while (true) {
        std::unique_lock<std::mutex> lk(g_sm);
        g_cv.wait(lk, [] { return g_stop || g_pending != -1; });
        if (g_stop) return;
        int L = g_pending; g_pending = -1;
        lk.unlock();
        if (L >= 0 && L < N_LAYERS) layer_madvise(L, MADV_WILLNEED);
    }
}
static void request_prefetch(int L) {
    std::lock_guard<std::mutex> lk(g_sm);
    g_pending = L; g_cv.notify_one();
}

// ---- linear: Y[M,N] = X[M,K] @ W^T, W row-major [N,K] ----
#if defined(__x86_64__) || defined(__i386__)
static inline float hsum256(__m256 v);
static inline float hmax256(__m256 v);
#endif
static void linear(const float* X, int M, int K, const float* W, int N, float* Y, int nt) {
#if defined(__x86_64__) || defined(__i386__)
    if (M == 1) {
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int j = 0; j < N; j++) {
            const float* wj = W + (size_t)j * K;
            __m256 acc = _mm256_setzero_ps();
            for (int k = 0; k < K; k += 8) {
                __m256 vx = _mm256_loadu_ps(X + k);
                __m256 vw = _mm256_loadu_ps(wj + k);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(vx, vw));
            }
            Y[j] = hsum256(acc);
        }
    } else {
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int i = 0; i < M; i++) {
            const float* xi = X + (size_t)i * K;
            float* yi = Y + (size_t)i * N;
            for (int j = 0; j < N; j++) {
                const float* wj = W + (size_t)j * K;
                __m256 acc = _mm256_setzero_ps();
                for (int k = 0; k < K; k += 8) {
                    __m256 vx = _mm256_loadu_ps(xi + k);
                    __m256 vw = _mm256_loadu_ps(wj + k);
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(vx, vw));
                }
                yi[j] = hsum256(acc);
            }
        }
    }
#else
    if (M == 1) {
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int j = 0; j < N; j++) {
            const float* wj = W + (size_t)j * K;
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += X[k] * wj[k];
            Y[j] = s;
        }
    } else {
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int i = 0; i < M; i++) {
            const float* xi = X + (size_t)i * K;
            float* yi = Y + (size_t)i * N;
            for (int j = 0; j < N; j++) {
                const float* wj = W + (size_t)j * K;
                float s = 0.0f;
                for (int k = 0; k < K; k++) s += xi[k] * wj[k];
                yi[j] = s;
            }
        }
    }
#endif
}

// ---- branchless AVX2 primitives ----
#if defined(__x86_64__) || defined(__i386__)

// horizontal sum of 8 floats
static inline float hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
// horizontal max of 8 floats
static inline float hmax256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2,3,0,1)));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1,0,3,2)));
    return _mm_cvtss_f32(m);
}

#endif

// ---- AVX2 INT8 GEMM: X (float) @ Wq (int8)^T, per-row dynamic activation quant ----
// Y[i,j] = sx_i * S[j] * sum_k qx_i[k] * Wq[j,k],  qx = round(X / sx), sx = maxabs(X)/127
// K must be a multiple of 16 (both HIDDEN=896 and INTER=4864 satisfy this).
#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2")))
static void linear_i8_avx2(const float* X, int M, int K, const int8_t* W, const float* S, int N, float* Y, int nt) {
    for (int i = 0; i < M; i++) {
        const float* xi = X + (size_t)i * K;
        // 1. Compute scale: mx = max(abs(xi)) / 127
        __m256 vmax = _mm256_setzero_ps();
        for (int k = 0; k < K; k += 8) {
            __m256 vx = _mm256_loadu_ps(xi + k);
            __m256 va = _mm256_max_ps(vx, _mm256_sub_ps(_mm256_setzero_ps(), vx));
            vmax = _mm256_max_ps(vmax, va);
        }
        float mx = hmax256(vmax);
        float sx = (mx > 0.0f) ? mx / 127.0f : 1.0f;
        float inv_sx = 1.0f / sx;
        // 2. Quantize: g_qx[k] = clamp(round(xi[k] * inv_sx), -127, 127)
        // Process 4 floats at a time -> 4 int8 using SSE (avoids lane-crossing pack issues)
        __m128 vsx128 = _mm_set1_ps(inv_sx);
        __m128i v127_128 = _mm_set1_epi32(127);
        __m128i vm127_128 = _mm_set1_epi32(-127);
        for (int k = 0; k < K; k += 4) {
            __m128 v = _mm_round_ps(_mm_mul_ps(_mm_loadu_ps(xi + k), vsx128),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m128i i32 = _mm_cvtps_epi32(v);
            i32 = _mm_min_epi32(v127_128, _mm_max_epi32(vm127_128, i32));
            __m128i i16 = _mm_packs_epi32(i32, _mm_setzero_si128());
            __m128i i8 = _mm_packs_epi16(i16, _mm_setzero_si128());
            *(int32_t*)(g_qx + k) = _mm_cvtsi128_si32(i8);
        }
        // 3. GEMM: Y[j] = sx * S[j] * sum_k g_qx[k] * W[j,k]
        float* yi = Y + (size_t)i * N;
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int j = 0; j < N; j++) {
            const int8_t* wj = W + (size_t)j * K;
            __m256i acc = _mm256_setzero_si256();
            int k = 0;
            for (; k < K; k += 16) {
                __m256i a = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)&g_qx[k]));
                __m256i b = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)&wj[k]));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a, b));
            }
            int32_t s32 = 0;
            for (int t = 0; t < 8; t++) s32 += _mm256_extract_epi32(acc, t);
            yi[j] = sx * S[j] * (float)s32;
        }
    }
}
#endif

// AVX2 forward declarations (used by rmsnorm, silu, add before their definitions)
#if defined(__x86_64__) || defined(__i386__)
static void rmsnorm_avx2(float* o, const float* x, const float* w, int n);
static void silu_mul_avx2(float* act, const float* gate, const float* up, int n);
static void add_avx2(float* a, const float* b, int n);
#endif

static void rmsnorm(float* o, const float* x, const float* w, int n) {
#if defined(__x86_64__) || defined(__i386__)
    rmsnorm_avx2(o, x, w, n);
#else
    float ss = 0.0f;
    #pragma omp parallel for reduction(+:ss)
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + EPS);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) o[i] = x[i] * r * w[i];
#endif
}

static inline float silu(float z) { return z / (1.0f + expf(-z)); }

// ---- branchless AVX2 primitives (continued) ----
#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2")))
static void rmsnorm_avx2(float* o, const float* x, const float* w, int n) {
    __m256 vss = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vss = _mm256_add_ps(vss, _mm256_mul_ps(vx, vx));
    }
    float ss = hsum256(vss);
    float r = 1.0f / sqrtf(ss / (float)n + EPS);
    __m256 vr = _mm256_set1_ps(r);
    for (int i = 0; i < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        _mm256_storeu_ps(o + i, _mm256_mul_ps(_mm256_mul_ps(vx, vr), vw));
    }
}

// fast approximate exp for AVX2 (relative error ~1e-4 over [-10, 10])
// based on: exp(x) ≈ 2^(x * log2(e)) with bit manipulation
__attribute__((target("avx2")))
static __m256 expf256_approx(__m256 x) {
    __m256 ln2 = _mm256_set1_ps(0.69314718056f);
    __m256 k = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504089f)); // x / ln2
    k = _mm256_round_ps(k, _MM_FROUND_TO_NEAREST_INT);
    __m256 t = _mm256_sub_ps(x, _mm256_mul_ps(k, ln2));
    // p(t) = 1 + t + t^2/2 + t^3/6 + t^4/24 + t^5/120
    __m256 p = _mm256_set1_ps(1.0f);
    p = _mm256_add_ps(p, _mm256_mul_ps(t, _mm256_set1_ps(1.0f)));
    __m256 t2 = _mm256_mul_ps(t, t);
    p = _mm256_add_ps(p, _mm256_mul_ps(t2, _mm256_set1_ps(0.5f)));
    __m256 t3 = _mm256_mul_ps(t2, t);
    p = _mm256_add_ps(p, _mm256_mul_ps(t3, _mm256_set1_ps(1.0f/6.0f)));
    __m256 t4 = _mm256_mul_ps(t2, t2);
    p = _mm256_add_ps(p, _mm256_mul_ps(t4, _mm256_set1_ps(1.0f/24.0f)));
    __m256 t5 = _mm256_mul_ps(t4, t);
    p = _mm256_add_ps(p, _mm256_mul_ps(t5, _mm256_set1_ps(1.0f/120.0f)));
    // 2^k via reinterpret
    __m256i ki = _mm256_cvtps_epi32(k);
    __m256i expo = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(expo));
}

__attribute__((target("avx2")))
static void silu_mul_avx2(float* act, const float* gate, const float* up, int n) {
    __m256 one = _mm256_set1_ps(1.0f);
    for (int i = 0; i < n; i += 8) {
        __m256 vg = _mm256_loadu_ps(gate + i);
        __m256 vu = _mm256_loadu_ps(up + i);
        __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), vg);
        __m256 ve = expf256_approx(neg);
        __m256 vs = _mm256_div_ps(vg, _mm256_add_ps(one, ve));
        _mm256_storeu_ps(act + i, _mm256_mul_ps(vs, vu));
    }
}

__attribute__((target("avx2")))
static void add_avx2(float* a, const float* b, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(a + i, _mm256_add_ps(va, vb));
    }
}

// AVX2 dot product helpers for lm_head (8-wide, no remainder — HIDDEN=896 is multiple of 8)
__attribute__((target("avx2")))
static inline float dot_xn_bf16_avx2(const float* xn, const uint16_t* r, int n) {
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < n; k += 8) {
        __m128i bf16 = _mm_loadu_si128((const __m128i*)&r[k]);
        __m256i i32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16), 16);
        __m256 vw = _mm256_castsi256_ps(i32);
        __m256 vx = _mm256_loadu_ps(xn + k);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(vx, vw));
    }
    return hsum256(acc);
}

__attribute__((target("avx2")))
static inline float dot_xn_i8_avx2(const float* xn, const int8_t* r, int n) {
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < n; k += 8) {
        __m256i vi8 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)&r[k]));
        __m256 vf = _mm256_cvtepi32_ps(vi8);
        __m256 vx = _mm256_loadu_ps(xn + k);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(vx, vf));
    }
    return hsum256(acc);
}

// Q8_0 dot product for GGUF lm_head: row points to Q8_0 blocks, n = HIDDEN
__attribute__((target("avx2")))
static inline float dot_xn_q8_avx2(const float* xn, const void* row, int n) {
    const uint8_t* blk = (const uint8_t*)row;
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < n; k += 32) {
        float d = fp16_to_f32(*(const uint16_t*)blk);
        __m256 vd = _mm256_set1_ps(d);
        const int8_t* qs = (const int8_t*)(blk + 2);
        for (int t = 0; t < 32; t += 8) {
            __m256i vi8 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)&qs[t]));
            __m256 vf = _mm256_mul_ps(_mm256_cvtepi32_ps(vi8), vd);
            __m256 vx = _mm256_loadu_ps(xn + k + t);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(vx, vf));
        }
        blk += 34;
    }
    return hsum256(acc);
}
// F32 dot product for GGUF lm_head
__attribute__((target("avx2")))
static inline float dot_xn_f32_avx2(const float* xn, const float* row, int n) {
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < n; k += 8) {
        __m256 vx = _mm256_loadu_ps(xn + k);
        __m256 vw = _mm256_loadu_ps(row + k);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(vx, vw));
    }
    return hsum256(acc);
}
#endif

static void rope(float* vec, int n_heads, int pos) {
    for (int h = 0; h < n_heads; h++) {
        float* base = vec + h * HEAD_DIM;
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            float ang = g_inv_freq[i] * (float)pos;
            float c = cosf(ang), s = sinf(ang);
            float x0 = base[i];
            float x1 = base[i + HEAD_DIM / 2];
            base[i] = x0 * c - x1 * s;
            base[i + HEAD_DIM / 2] = x0 * s + x1 * c;
        }
    }
}

// forward one token at position `pos` through `max_layers` layers.
// After the last layer, always does final norm + lm_head (into logits).
// If max_layers < N_LAYERS, this is a draft forward (early exit).
static void forward_token_(int token, int pos, int nt, int max_layers) {
    // embed
    if (g_mode == 2) {
        size_t row_off = (size_t)token * g_emb_gguf_row_bytes;
        const char* row = (const char*)g_emb_gguf + row_off;
        if (g_emb_gguf_type == GGML_BF16) {
            const uint16_t* r = (const uint16_t*)row;
            for (int i = 0; i < HIDDEN; i++) x_buf[i] = bf16_to_f32(r[i]);
        } else if (g_emb_gguf_type == GGML_F32) {
            memcpy(x_buf, row, HIDDEN * 4);
        } else if (g_emb_gguf_type == GGML_Q8_0) {
            int n_blocks = (HIDDEN + 31) / 32;
            for (int b = 0; b < n_blocks; b++) {
                float d = fp16_to_f32(*(const uint16_t*)(row + b * 34));
                const int8_t* qs = (const int8_t*)(row + b * 34 + 2);
                int rem = HIDDEN - b * 32;
                int bn = rem > 32 ? 32 : rem;
                for (int i = 0; i < bn; i++) x_buf[b * 32 + i] = d * (float)qs[i];
            }
        } else if (g_emb_gguf_type == GGML_F16) {
            const uint16_t* r = (const uint16_t*)row;
            for (int i = 0; i < HIDDEN; i++) x_buf[i] = fp16_to_f32(r[i]);
        }
    } else if (g_mode == 0) {
        const uint16_t* row = g_emb + (size_t)token * HIDDEN;
        for (int i = 0; i < HIDDEN; i++) x_buf[i] = bf16_to_f32(row[i]);
    } else {
        const int8_t* row = g_emb_i8 + (size_t)token * HIDDEN;
        float sv = g_emb_scale[token];
        for (int i = 0; i < HIDDEN; i++) x_buf[i] = sv * (float)row[i];
    }

    for (int L = 0; L < max_layers; L++) {
        // async: while we compute layer L, the worker prefetches layer L+1
        request_prefetch(L + 1);
        // --- self-attention ---
        rmsnorm(xn_buf, x_buf, norm_weight("model.layers." + std::to_string(L) + ".input_layernorm.weight"), HIDDEN);
        proj("model.layers." + std::to_string(L) + ".self_attn.q_proj.weight",
             xn_buf, 1, HIDDEN, N_HEADS * HEAD_DIM, q_buf, nt,
             "model.layers." + std::to_string(L) + ".self_attn.q_proj.bias");
        proj("model.layers." + std::to_string(L) + ".self_attn.k_proj.weight",
             xn_buf, 1, HIDDEN, N_KV * HEAD_DIM, k_buf, nt,
             "model.layers." + std::to_string(L) + ".self_attn.k_proj.bias");
        proj("model.layers." + std::to_string(L) + ".self_attn.v_proj.weight",
             xn_buf, 1, HIDDEN, N_KV * HEAD_DIM, v_buf, nt,
             "model.layers." + std::to_string(L) + ".self_attn.v_proj.bias");
        rope(q_buf, N_HEADS, pos);
        rope(k_buf, N_KV, pos);

        // append K,V to cache
        float* kc = g_kcache + ((size_t)L * MAX_SEQ + pos) * N_KV * HEAD_DIM;
        float* vc = g_vcache + ((size_t)L * MAX_SEQ + pos) * N_KV * HEAD_DIM;
        std::memcpy(kc, k_buf, N_KV * HEAD_DIM * sizeof(float));
        std::memcpy(vc, v_buf, N_KV * HEAD_DIM * sizeof(float));

        // attention
        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        for (int h = 0; h < N_HEADS; h++) {
            int g = h / N_GROUPS;
            float* qh = q_buf + h * HEAD_DIM;
            float* kc_g = g_kcache + ((size_t)L * MAX_SEQ) * N_KV * HEAD_DIM + (size_t)g * HEAD_DIM;
            float* vc_g = g_vcache + ((size_t)L * MAX_SEQ) * N_KV * HEAD_DIM + (size_t)g * HEAD_DIM;
            float maxs = -1e30f;
            for (int p = 0; p <= pos; p++) {
                const float* kp = kc_g + (size_t)p * N_KV * HEAD_DIM;
                float s = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) s += qh[d] * kp[d];
                s *= scale;
                scores[p] = s;
                if (s > maxs) maxs = s;
            }
            float sum = 0.0f;
            for (int p = 0; p <= pos; p++) { scores[p] = expf(scores[p] - maxs); sum += scores[p]; }
            float inv = 1.0f / sum;
            float* ch = ctx_buf + h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) ch[d] = 0.0f;
            for (int p = 0; p <= pos; p++) {
                const float* vp = vc_g + (size_t)p * N_KV * HEAD_DIM;
                float w = scores[p] * inv;
                for (int d = 0; d < HEAD_DIM; d++) ch[d] += w * vp[d];
            }
        }
        proj("model.layers." + std::to_string(L) + ".self_attn.o_proj.weight",
             ctx_buf, 1, HIDDEN, HIDDEN, attn_out, nt);
#if defined(__x86_64__) || defined(__i386__)
        add_avx2(x_buf, attn_out, HIDDEN);
#else
        for (int i = 0; i < HIDDEN; i++) x_buf[i] += attn_out[i];
#endif

        // --- mlp ---
        rmsnorm(xn_buf, x_buf, norm_weight("model.layers." + std::to_string(L) + ".post_attention_layernorm.weight"), HIDDEN);
        proj("model.layers." + std::to_string(L) + ".mlp.gate_proj.weight",
             xn_buf, 1, HIDDEN, INTER, gate_buf, nt);
        proj("model.layers." + std::to_string(L) + ".mlp.up_proj.weight",
             xn_buf, 1, HIDDEN, INTER, up_buf, nt);
#if defined(__x86_64__) || defined(__i386__)
        silu_mul_avx2(act_buf, gate_buf, up_buf, INTER);
#else
        for (int i = 0; i < INTER; i++) act_buf[i] = silu(gate_buf[i]) * up_buf[i];
#endif
        proj("model.layers." + std::to_string(L) + ".mlp.down_proj.weight",
             act_buf, 1, INTER, HIDDEN, ffn_out, nt);
#if defined(__x86_64__) || defined(__i386__)
        add_avx2(x_buf, ffn_out, HIDDEN);
#else
        for (int i = 0; i < HIDDEN; i++) x_buf[i] += ffn_out[i];
#endif

        // async: release this layer's pages from the OS page-cache (sliding window)
        layer_madvise(L, MADV_DONTNEED);
    }

    // final norm
    rmsnorm(xn_buf, x_buf, norm_weight("model.norm.weight"), HIDDEN);

    // lm_head = tie(embed_tokens): logits[v] = dot(xn, emb[v])  (tiled streaming)
    const int BLOCK = 4096;
    #pragma omp parallel for schedule(dynamic) num_threads(nt)
    for (int v0 = 0; v0 < VOCAB; v0 += BLOCK) {
        int b = (v0 + BLOCK > VOCAB) ? (VOCAB - v0) : BLOCK;
        if (g_mode == 2) {
#if defined(__x86_64__) || defined(__i386__)
            if (g_emb_gguf_type == GGML_BF16) {
                for (int j = 0; j < b; j++) {
                    const uint16_t* r = (const uint16_t*)((const char*)g_emb_gguf + (size_t)(v0 + j) * g_emb_gguf_row_bytes);
                    logits[v0 + j] = dot_xn_bf16_avx2(xn_buf, r, HIDDEN);
                }
            } else if (g_emb_gguf_type == GGML_F32) {
                for (int j = 0; j < b; j++) {
                    const float* r = (const float*)((const char*)g_emb_gguf + (size_t)(v0 + j) * g_emb_gguf_row_bytes);
                    logits[v0 + j] = dot_xn_f32_avx2(xn_buf, r, HIDDEN);
                }
            } else if (g_emb_gguf_type == GGML_Q8_0) {
                for (int j = 0; j < b; j++) {
                    const void* r = (const char*)g_emb_gguf + (size_t)(v0 + j) * g_emb_gguf_row_bytes;
                    logits[v0 + j] = dot_xn_q8_avx2(xn_buf, r, HIDDEN);
                }
            } else if (g_emb_gguf_type == GGML_F16) {
                for (int j = 0; j < b; j++) {
                    const uint16_t* r = (const uint16_t*)((const char*)g_emb_gguf + (size_t)(v0 + j) * g_emb_gguf_row_bytes);
                    logits[v0 + j] = dot_xn_bf16_avx2(xn_buf, r, HIDDEN);
                }
            }
#else
            for (int j = 0; j < b; j++) {
                const char* row = (const char*)g_emb_gguf + (size_t)(v0 + j) * g_emb_gguf_row_bytes;
                float s = 0.0f;
                if (g_emb_gguf_type == GGML_BF16 || g_emb_gguf_type == GGML_F16) {
                    for (int k = 0; k < HIDDEN; k++) {
                        uint16_t v = ((const uint16_t*)row)[k];
                        s += xn_buf[k] * (g_emb_gguf_type == GGML_BF16 ? bf16_to_f32(v) : fp16_to_f32(v));
                    }
                } else if (g_emb_gguf_type == GGML_F32) {
                    for (int k = 0; k < HIDDEN; k++) s += xn_buf[k] * ((const float*)row)[k];
                } else if (g_emb_gguf_type == GGML_Q8_0) {
                    int n_blocks = (HIDDEN + 31) / 32;
                    for (int bi = 0; bi < n_blocks; bi++) {
                        float d = fp16_to_f32(*(const uint16_t*)(row + bi * 34));
                        const int8_t* qs = (const int8_t*)(row + bi * 34 + 2);
                        int rem = HIDDEN - bi * 32;
                        int bn = rem > 32 ? 32 : rem;
                        for (int i = 0; i < bn; i++) s += xn_buf[bi * 32 + i] * d * (float)qs[i];
                    }
                }
                logits[v0 + j] = s;
            }
#endif
        } else if (g_mode == 0) {
#if defined(__x86_64__) || defined(__i386__)
            for (int j = 0; j < b; j++) {
                const uint16_t* r = g_emb + (size_t)(v0 + j) * HIDDEN;
                logits[v0 + j] = dot_xn_bf16_avx2(xn_buf, r, HIDDEN);
            }
#else
            for (int j = 0; j < b; j++) {
                const uint16_t* r = g_emb + (size_t)(v0 + j) * HIDDEN;
                float s = 0.0f;
                for (int k = 0; k < HIDDEN; k++) s += xn_buf[k] * bf16_to_f32(r[k]);
                logits[v0 + j] = s;
            }
#endif
        } else {
#if defined(__x86_64__) || defined(__i386__)
            for (int j = 0; j < b; j++) {
                const int8_t* r = g_emb_i8 + (size_t)(v0 + j) * HIDDEN;
                logits[v0 + j] = g_emb_scale[v0 + j] * dot_xn_i8_avx2(xn_buf, r, HIDDEN);
            }
#else
            for (int j = 0; j < b; j++) {
                const int8_t* r = g_emb_i8 + (size_t)(v0 + j) * HIDDEN;
                float sv = g_emb_scale[v0 + j];
                float s = 0.0f;
                for (int k = 0; k < HIDDEN; k++) s += xn_buf[k] * (float)r[k];
                logits[v0 + j] = sv * s;
            }
#endif
        }
    }
}

static void forward_token(int token, int pos, int nt) {
    forward_token_(token, pos, nt, N_LAYERS);
}

// ---- forward_block: batched M-token forward through all N layers ----
// tokens[M] — input token IDs, one per position
// pos — absolute start position
// After completion, logits contains predictions for the LAST token (pos+M-1).
// KV cache is updated for all N layers at positions pos..pos+M-1.
static void forward_block(const int* tokens, int M, int pos, int nt) {
    // embed all M tokens into x_block
    for (int i = 0; i < M; i++) {
        int tok = tokens[i];
        float* xi = x_block + (size_t)i * HIDDEN;
        if (g_mode == 2) {
            size_t row_off = (size_t)tok * g_emb_gguf_row_bytes;
            const char* row = (const char*)g_emb_gguf + row_off;
            if (g_emb_gguf_type == GGML_BF16) {
                const uint16_t* r = (const uint16_t*)row;
                for (int j = 0; j < HIDDEN; j++) xi[j] = bf16_to_f32(r[j]);
            } else if (g_emb_gguf_type == GGML_F32) {
                memcpy(xi, row, HIDDEN * 4);
            } else if (g_emb_gguf_type == GGML_Q8_0) {
                int n_blocks = (HIDDEN + 31) / 32;
                for (int b = 0; b < n_blocks; b++) {
                    float d = fp16_to_f32(*(const uint16_t*)(row + b * 34));
                    const int8_t* qs = (const int8_t*)(row + b * 34 + 2);
                    int rem = HIDDEN - b * 32;
                    int bn = rem > 32 ? 32 : rem;
                    for (int j = 0; j < bn; j++) xi[b * 32 + j] = d * (float)qs[j];
                }
            } else if (g_emb_gguf_type == GGML_F16) {
                const uint16_t* r = (const uint16_t*)row;
                for (int j = 0; j < HIDDEN; j++) xi[j] = fp16_to_f32(r[j]);
            }
        } else if (g_mode == 0) {
            const uint16_t* row = g_emb + (size_t)tok * HIDDEN;
            for (int j = 0; j < HIDDEN; j++) xi[j] = bf16_to_f32(row[j]);
        } else {
            const int8_t* row = g_emb_i8 + (size_t)tok * HIDDEN;
            float sv = g_emb_scale[tok];
            for (int j = 0; j < HIDDEN; j++) xi[j] = sv * (float)row[j];
        }
    }

    for (int L = 0; L < N_LAYERS; L++) {
        request_prefetch(L + 1);

        // batch RMSNorm
        const float* nw = norm_weight(std::string("model.layers." + std::to_string(L) + ".input_layernorm.weight"));
        #pragma omp parallel for num_threads(nt)
        for (int i = 0; i < M; i++) {
            rmsnorm(xn_block_b + (size_t)i * HIDDEN, x_block + (size_t)i * HIDDEN, nw, HIDDEN);
        }

        // batch QKV projections
        proj("model.layers." + std::to_string(L) + ".self_attn.q_proj.weight",
             xn_block_b, M, HIDDEN, N_HEADS * HEAD_DIM, q_block_b, nt,
             "model.layers." + std::to_string(L) + ".self_attn.q_proj.bias");
        proj("model.layers." + std::to_string(L) + ".self_attn.k_proj.weight",
             xn_block_b, M, HIDDEN, N_KV * HEAD_DIM, k_block_b, nt,
             "model.layers." + std::to_string(L) + ".self_attn.k_proj.bias");
        proj("model.layers." + std::to_string(L) + ".self_attn.v_proj.weight",
             xn_block_b, M, HIDDEN, N_KV * HEAD_DIM, v_block_b, nt,
             "model.layers." + std::to_string(L) + ".self_attn.v_proj.bias");

        // load O_proj weights once
        const std::string o_name = "model.layers." + std::to_string(L) + ".self_attn.o_proj.weight";
        float* o_w = nullptr;
        const WT* o_wt = nullptr;
        bool o_loaded = false;
        if (g_mode == 0) { load_weight(o_name, weight_buf); o_w = weight_buf; o_loaded = true; }
        else if (g_mode == 2) { load_weight_gguf(o_name, weight_buf); o_w = weight_buf; o_loaded = true; }
        else o_wt = &g_w[o_name];

        // per-token: RoPE, KV store, attention, O_proj, residual
        for (int i = 0; i < M; i++) {
            float* qi = q_block_b + (size_t)i * HIDDEN;
            float* ki = k_block_b + (size_t)i * N_KV * HEAD_DIM;
            float* vi = v_block_b + (size_t)i * N_KV * HEAD_DIM;
            int p = pos + i;

            rope(qi, N_HEADS, p);
            rope(ki, N_KV, p);

            float* kc = g_kcache + ((size_t)L * MAX_SEQ + p) * N_KV * HEAD_DIM;
            float* vc = g_vcache + ((size_t)L * MAX_SEQ + p) * N_KV * HEAD_DIM;
            memcpy(kc, ki, (size_t)N_KV * HEAD_DIM * sizeof(float));
            memcpy(vc, vi, (size_t)N_KV * HEAD_DIM * sizeof(float));

            // causal attention over 0..p
            float scale = 1.0f / sqrtf((float)HEAD_DIM);
            for (int h = 0; h < N_HEADS; h++) {
                int g = h / N_GROUPS;
                float* qh = qi + h * HEAD_DIM;
                float* kc_g = g_kcache + ((size_t)L * MAX_SEQ) * N_KV * HEAD_DIM + (size_t)g * HEAD_DIM;
                float* vc_g = g_vcache + ((size_t)L * MAX_SEQ) * N_KV * HEAD_DIM + (size_t)g * HEAD_DIM;
                float maxs = -1e30f;
                for (int pp = 0; pp <= p; pp++) {
                    const float* kp = kc_g + (size_t)pp * N_KV * HEAD_DIM;
                    float s = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++) s += qh[d] * kp[d];
                    s *= scale;
                    scores[pp] = s;
                    if (s > maxs) maxs = s;
                }
                float sum = 0.0f;
                for (int pp = 0; pp <= p; pp++) { scores[pp] = expf(scores[pp] - maxs); sum += scores[pp]; }
                float inv = 1.0f / sum;
                float* ch = ctx_buf + h * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; d++) ch[d] = 0.0f;
                for (int pp = 0; pp <= p; pp++) {
                    const float* vp = vc_g + (size_t)pp * N_KV * HEAD_DIM;
                    float w = scores[pp] * inv;
                    for (int d = 0; d < HEAD_DIM; d++) ch[d] += w * vp[d];
                }
            }

            // O_proj (weights already loaded)
            if (g_mode == 0 || g_mode == 2) linear(ctx_buf, 1, HIDDEN, o_w, HIDDEN, attn_out, nt);
            else linear_i8_avx2(ctx_buf, 1, HIDDEN, (const int8_t*)o_wt->data, o_wt->scale, HIDDEN, attn_out, nt);

#if defined(__x86_64__) || defined(__i386__)
            add_avx2(x_block + (size_t)i * HIDDEN, attn_out, HIDDEN);
#else
            for (int d = 0; d < HIDDEN; d++) x_block[(size_t)i * HIDDEN + d] += attn_out[d];
#endif
        }

        // post-attention norm + MLP (batched)
        const float* pnw = norm_weight(std::string("model.layers." + std::to_string(L) + ".post_attention_layernorm.weight"));
        #pragma omp parallel for num_threads(nt)
        for (int i = 0; i < M; i++) {
            rmsnorm(xn_block_b + (size_t)i * HIDDEN, x_block + (size_t)i * HIDDEN, pnw, HIDDEN);
        }

        proj("model.layers." + std::to_string(L) + ".mlp.gate_proj.weight",
             xn_block_b, M, HIDDEN, INTER, gate_block_b, nt);
        proj("model.layers." + std::to_string(L) + ".mlp.up_proj.weight",
             xn_block_b, M, HIDDEN, INTER, up_block_b, nt);

#if defined(__x86_64__) || defined(__i386__)
        silu_mul_avx2(act_block_b, gate_block_b, up_block_b, M * INTER);
#else
        #pragma omp parallel for num_threads(nt)
        for (int i = 0; i < M * INTER; i++) {
            float g = gate_block_b[i];
            act_block_b[i] = (g / (1.0f + expf(-g))) * up_block_b[i];
        }
#endif

        proj("model.layers." + std::to_string(L) + ".mlp.down_proj.weight",
             act_block_b, M, INTER, HIDDEN, ffn_block_b, nt);

        for (int i = 0; i < M; i++) {
#if defined(__x86_64__) || defined(__i386__)
            add_avx2(x_block + (size_t)i * HIDDEN, ffn_block_b + (size_t)i * HIDDEN, HIDDEN);
#else
            float* xi = x_block + (size_t)i * HIDDEN;
            float* fi = ffn_block_b + (size_t)i * HIDDEN;
            for (int d = 0; d < HIDDEN; d++) xi[d] += fi[d];
#endif
        }

        layer_madvise(L, MADV_DONTNEED);
    }

    // final norm for LAST token only (next-token prediction)
    rmsnorm(xn_buf, x_block + (size_t)(M - 1) * HIDDEN, norm_weight("model.norm.weight"), HIDDEN);

    // lm_head for last position
    const int BLOCK = 4096;
    #pragma omp parallel for schedule(dynamic) num_threads(nt)
    for (int v0 = 0; v0 < VOCAB; v0 += BLOCK) {
        int b = (v0 + BLOCK > VOCAB) ? (VOCAB - v0) : BLOCK;
        if (g_mode == 2) {
#if defined(__x86_64__) || defined(__i386__)
            if (g_emb_gguf_type == GGML_BF16) {
                for (int j = 0; j < b; j++) {
                    const uint16_t* r = (const uint16_t*)((const char*)g_emb_gguf + (size_t)(v0 + j) * g_emb_gguf_row_bytes);
                    logits[v0 + j] = dot_xn_bf16_avx2(xn_buf, r, HIDDEN);
                }
            } else if (g_emb_gguf_type == GGML_F32) {
                for (int j = 0; j < b; j++) {
                    const float* r = (const float*)((const char*)g_emb_gguf + (size_t)(v0 + j) * g_emb_gguf_row_bytes);
                    logits[v0 + j] = dot_xn_f32_avx2(xn_buf, r, HIDDEN);
                }
            } else if (g_emb_gguf_type == GGML_Q8_0) {
                for (int j = 0; j < b; j++) {
                    const void* r = (const char*)g_emb_gguf + (size_t)(v0 + j) * g_emb_gguf_row_bytes;
                    logits[v0 + j] = dot_xn_q8_avx2(xn_buf, r, HIDDEN);
                }
            } else if (g_emb_gguf_type == GGML_F16) {
                for (int j = 0; j < b; j++) {
                    const uint16_t* r = (const uint16_t*)((const char*)g_emb_gguf + (size_t)(v0 + j) * g_emb_gguf_row_bytes);
                    logits[v0 + j] = dot_xn_bf16_avx2(xn_buf, r, HIDDEN);
                }
            }
#else
            for (int j = 0; j < b; j++) {
                const char* row = (const char*)g_emb_gguf + (size_t)(v0 + j) * g_emb_gguf_row_bytes;
                float s = 0.0f;
                if (g_emb_gguf_type == GGML_BF16 || g_emb_gguf_type == GGML_F16) {
                    for (int k = 0; k < HIDDEN; k++) {
                        uint16_t v = ((const uint16_t*)row)[k];
                        s += xn_buf[k] * (g_emb_gguf_type == GGML_BF16 ? bf16_to_f32(v) : fp16_to_f32(v));
                    }
                } else if (g_emb_gguf_type == GGML_F32) {
                    for (int k = 0; k < HIDDEN; k++) s += xn_buf[k] * ((const float*)row)[k];
                } else if (g_emb_gguf_type == GGML_Q8_0) {
                    int n_blocks = (HIDDEN + 31) / 32;
                    for (int bi = 0; bi < n_blocks; bi++) {
                        float d = fp16_to_f32(*(const uint16_t*)(row + bi * 34));
                        const int8_t* qs = (const int8_t*)(row + bi * 34 + 2);
                        int rem = HIDDEN - bi * 32;
                        int bn = rem > 32 ? 32 : rem;
                        for (int i = 0; i < bn; i++) s += xn_buf[bi * 32 + i] * d * (float)qs[i];
                    }
                }
                logits[v0 + j] = s;
            }
#endif
        } else if (g_mode == 0) {
#if defined(__x86_64__) || defined(__i386__)
            for (int j = 0; j < b; j++) {
                const uint16_t* r = g_emb + (size_t)(v0 + j) * HIDDEN;
                logits[v0 + j] = dot_xn_bf16_avx2(xn_buf, r, HIDDEN);
            }
#else
            for (int j = 0; j < b; j++) {
                const uint16_t* r = g_emb + (size_t)(v0 + j) * HIDDEN;
                float s = 0.0f;
                for (int k = 0; k < HIDDEN; k++) s += xn_buf[k] * bf16_to_f32(r[k]);
                logits[v0 + j] = s;
            }
#endif
        } else {
#if defined(__x86_64__) || defined(__i386__)
            for (int j = 0; j < b; j++) {
                const int8_t* r = g_emb_i8 + (size_t)(v0 + j) * HIDDEN;
                logits[v0 + j] = g_emb_scale[v0 + j] * dot_xn_i8_avx2(xn_buf, r, HIDDEN);
            }
#else
            for (int j = 0; j < b; j++) {
                const int8_t* r = g_emb_i8 + (size_t)(v0 + j) * HIDDEN;
                float sv = g_emb_scale[v0 + j];
                float s = 0.0f;
                for (int k = 0; k < HIDDEN; k++) s += xn_buf[k] * (float)r[k];
                logits[v0 + j] = sv * s;
            }
#endif
        }
    }
}

// ---- lm_head for a single normalized state ----
static void lm_head(const float* xn, float* out, int nt) {
    const int BLOCK = 4096;
    #pragma omp parallel for schedule(dynamic) num_threads(nt)
    for (int v0 = 0; v0 < VOCAB; v0 += BLOCK) {
        int b = (v0 + BLOCK > VOCAB) ? (VOCAB - v0) : BLOCK;
        if (g_mode == 0) {
#if defined(__x86_64__) || defined(__i386__)
            for (int j = 0; j < b; j++) {
                const uint16_t* r = g_emb + (size_t)(v0 + j) * HIDDEN;
                out[v0 + j] = dot_xn_bf16_avx2(xn, r, HIDDEN);
            }
#else
            for (int j = 0; j < b; j++) {
                const uint16_t* r = g_emb + (size_t)(v0 + j) * HIDDEN;
                float s = 0.0f;
                for (int k = 0; k < HIDDEN; k++) s += xn[k] * bf16_to_f32(r[k]);
                out[v0 + j] = s;
            }
#endif
        } else if (g_mode == 2) {
#if defined(__x86_64__) || defined(__i386__)
            if (g_emb_gguf_type == GGML_BF16) {
                for (int j = 0; j < b; j++) {
                    const uint16_t* r = (const uint16_t*)((const char*)g_emb_gguf + (size_t)(v0 + j) * g_emb_gguf_row_bytes);
                    out[v0 + j] = dot_xn_bf16_avx2(xn, r, HIDDEN);
                }
            } else if (g_emb_gguf_type == GGML_F32) {
                for (int j = 0; j < b; j++) {
                    const float* r = (const float*)((const char*)g_emb_gguf + (size_t)(v0 + j) * g_emb_gguf_row_bytes);
                    out[v0 + j] = dot_xn_f32_avx2(xn, r, HIDDEN);
                }
            } else if (g_emb_gguf_type == GGML_Q8_0) {
                for (int j = 0; j < b; j++) {
                    const void* r = (const char*)g_emb_gguf + (size_t)(v0 + j) * g_emb_gguf_row_bytes;
                    out[v0 + j] = dot_xn_q8_avx2(xn, r, HIDDEN);
                }
            } else if (g_emb_gguf_type == GGML_F16) {
                for (int j = 0; j < b; j++) {
                    const uint16_t* r = (const uint16_t*)((const char*)g_emb_gguf + (size_t)(v0 + j) * g_emb_gguf_row_bytes);
                    out[v0 + j] = dot_xn_bf16_avx2(xn, r, HIDDEN);
                }
            }
#else
            for (int j = 0; j < b; j++) {
                const char* row = (const char*)g_emb_gguf + (size_t)(v0 + j) * g_emb_gguf_row_bytes;
                float s = 0.0f;
                if (g_emb_gguf_type == GGML_BF16 || g_emb_gguf_type == GGML_F16) {
                    for (int k = 0; k < HIDDEN; k++) {
                        uint16_t v = ((const uint16_t*)row)[k];
                        s += xn[k] * (g_emb_gguf_type == GGML_BF16 ? bf16_to_f32(v) : fp16_to_f32(v));
                    }
                } else if (g_emb_gguf_type == GGML_F32) {
                    for (int k = 0; k < HIDDEN; k++) s += xn[k] * ((const float*)row)[k];
                } else if (g_emb_gguf_type == GGML_Q8_0) {
                    int n_blocks = (HIDDEN + 31) / 32;
                    for (int bi = 0; bi < n_blocks; bi++) {
                        float d = fp16_to_f32(*(const uint16_t*)(row + bi * 34));
                        const int8_t* qs = (const int8_t*)(row + bi * 34 + 2);
                        int rem = HIDDEN - bi * 32;
                        int bn = rem > 32 ? 32 : rem;
                        for (int i = 0; i < bn; i++) s += xn[bi * 32 + i] * d * (float)qs[i];
                    }
                }
                out[v0 + j] = s;
            }
#endif
        } else {
#if defined(__x86_64__) || defined(__i386__)
            for (int j = 0; j < b; j++) {
                const int8_t* r = g_emb_i8 + (size_t)(v0 + j) * HIDDEN;
                out[v0 + j] = g_emb_scale[v0 + j] * dot_xn_i8_avx2(xn, r, HIDDEN);
            }
#else
            for (int j = 0; j < b; j++) {
                const int8_t* r = g_emb_i8 + (size_t)(v0 + j) * HIDDEN;
                float sv = g_emb_scale[v0 + j];
                float s = 0.0f;
                for (int k = 0; k < HIDDEN; k++) s += xn[k] * (float)r[k];
                out[v0 + j] = sv * s;
            }
#endif
        }
    }
}

static int argmax(const float* a, int n) {
    int best = 0; float bv = a[0];
    for (int i = 1; i < n; i++) if (a[i] > bv) { bv = a[i]; best = i; }
    return best;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <prompt_tokens_file> [max_new_tokens]\n", argv[0]);
        fprintf(stderr, "  env DSO_MODEL: path to .safetensors (BF16) or .dso (INT8)\n");
        return 1;
    }
    const char* tok_path = argv[1];
    int max_new = (argc > 2) ? atoi(argv[2]) : 64;

    // mmap model
    const char* model_path = "/home/lain/dso/model/model.safetensors";
    const char* env_model = getenv("DSO_MODEL");
    if (env_model) model_path = env_model;
    if (strstr(model_path, ".dso") != nullptr) g_mode = 1;
    else if (strstr(model_path, ".gguf") != nullptr) g_mode = 2;
    else g_mode = 0;
    g_fd = open(model_path, O_RDONLY);
    if (g_fd < 0) { perror("open model"); return 1; }
    struct stat st; fstat(g_fd, &st); g_map_size = st.st_size;
    g_map = (char*)mmap(nullptr, g_map_size, PROT_READ, MAP_PRIVATE, g_fd, 0);
    if (g_map == MAP_FAILED) { perror("mmap"); return 1; }
    if (g_mode == 0) parse_header();
    else if (g_mode == 1) parse_dso();
    else parse_gguf();

    // async streaming: prefetch embedding matrix into page-cache, start worker
    madvise_tensor("model.embed_tokens.weight", MADV_WILLNEED);
    g_worker = std::thread(stream_worker);

    // alloc static KV cache once
    size_t kv_elems = (size_t)N_LAYERS * MAX_SEQ * N_KV * HEAD_DIM;
    g_kcache = (float*)malloc(kv_elems * sizeof(float));
    g_vcache = (float*)malloc(kv_elems * sizeof(float));

    // precompute rope inv_freq
    for (int i = 0; i < HEAD_DIM / 2; i++)
        g_inv_freq[i] = 1.0f / powf(ROPE_THETA, (2.0f * i) / HEAD_DIM);

    // read prompt token ids (space separated)
    FILE* pf = fopen(tok_path, "r");
    if (!pf) { perror("open prompt tokens"); return 1; }
    std::vector<int> prompt;
    int t;
    while (fscanf(pf, "%d", &t) == 1) prompt.push_back(t);
    fclose(pf);
    if (prompt.empty()) { fprintf(stderr, "empty prompt\n"); return 1; }

    int nt = omp_get_max_threads();
    fprintf(stderr, "[dso] threads=%d prompt_len=%zu max_new=%d\n", nt, prompt.size(), max_new);

    // ---- prefill (process prompt, build KV) ----
    forward_block(prompt.data(), (int)prompt.size(), 0, nt);
    int pos = (int)prompt.size();
    // first generated token from last logits
    int next = argmax(logits, VOCAB);
    std::vector<int> out;
    out.push_back(next);

    // ---- decode loop ----
    for (int step = 1; step < max_new; step++) {
        forward_token(next, pos, nt);
        pos++;
        next = argmax(logits, VOCAB);
        out.push_back(next);
        if (next == 151645 && !getenv("DSO_NOEOS")) break; // eos_token_id
    }

    // stop async worker
    {
        std::lock_guard<std::mutex> lk(g_sm);
        g_stop = true; g_cv.notify_one();
    }
    if (g_worker.joinable()) g_worker.join();

    // emit generated token ids
    for (size_t i = 0; i < out.size(); i++) printf("%d%c", out[i], (i+1<out.size()?' ':'\n'));
    return 0;
}
