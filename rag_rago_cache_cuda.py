# rag_rago_cache_cuda.py
import time
import hashlib
from collections import OrderedDict

import cupy as cp


class LittleLru:
    def __init__(self, cap):
        self.cap = cap
        self.box = OrderedDict()

    def get(self, key):
        if key not in self.box:
            return None
        val = self.box.pop(key)
        self.box[key] = val
        return val

    def put(self, key, val):
        if key in self.box:
            self.box.pop(key)
        self.box[key] = val
        if len(self.box) > self.cap:
            self.box.popitem(last=False)


def _fingerprint(txt: str) -> str:
    return hashlib.blake2b(txt.encode("utf-8"), digest_size=16).hexdigest()


def make_docs(n_docs=200_000, dim=256, seed=7):
    rs = cp.random.RandomState(seed)
    raw = rs.standard_normal((n_docs, dim), dtype=cp.float32)
    raw /= cp.linalg.norm(raw, axis=1, keepdims=True) + 1e-9
    return raw


def int8_pack(mat_fp16: cp.ndarray):
    # simple per-row scale, store int8
    mx = cp.max(cp.abs(mat_fp16), axis=1, keepdims=True) + 1e-8
    scale = (127.0 / mx).astype(cp.float16)
    packed = cp.clip(cp.rint(mat_fp16 * scale), -127, 127).astype(cp.int8)
    inv = (1.0 / scale).astype(cp.float16)
    return packed, inv


def int8_unpack(packed: cp.ndarray, inv_scale: cp.ndarray):
    return (packed.astype(cp.float16) * inv_scale).astype(cp.float16)


def gpu_retrieve(query_vecs, doc_bank_fp16, top_k):
    # brute-force cosine (dot since normalized)
    scores = query_vecs @ doc_bank_fp16.T
    best = cp.argpartition(scores, -top_k, axis=1)[:, -top_k:]
    best_scores = cp.take_along_axis(scores, best, axis=1)
    order = cp.argsort(best_scores, axis=1)[:, ::-1]
    best = cp.take_along_axis(best, order, axis=1)
    return best


def fake_context_builder(doc_ids, token_budget):
    # pretend each doc contributes ~45 tokens
    room = token_budget
    picked = []
    for d in doc_ids:
        if room < 45:
            break
        picked.append(int(d))
        room -= 45
    return picked


def fake_generate(question, chosen_docs, cheap_mode):
    # simulate time ~ tokens
    base = 0.012 if cheap_mode else 0.018
    extra = 0.00003 * (len(chosen_docs) * 45)
    time.sleep(base + extra)
    return f"Answer: {question} (docs={len(chosen_docs)})"


def choose_knobs(p95_budget_ms, last_retr_ms, last_gen_ms):
    # tiny heuristic “model”
    top_k = 10
    token_budget = 320
    cheap_mode = False
    use_int8_store = False

    if last_gen_ms > p95_budget_ms * 0.55:
        token_budget = 220
        top_k = 7
        cheap_mode = True

    if last_retr_ms > p95_budget_ms * 0.30:
        top_k = max(5, top_k - 2)
        use_int8_store = True

    return dict(top_k=top_k, token_budget=token_budget, cheap_mode=cheap_mode, use_int8_store=use_int8_store)


def main():
    cp.cuda.Device(0).use()

    doc_bank = make_docs(n_docs=120_000, dim=192)
    doc_bank_fp16 = doc_bank.astype(cp.float16)

    packed_i8, inv_scale = int8_pack(doc_bank_fp16)

    retr_cache = LittleLru(cap=512)

    traffic = [
        "what is rag latency?",
        "how to reduce rag cost?",
        "what is rag latency?",
        "explain caching in rag",
        "how to reduce rag cost?",
        "what is ragserve vs rago?",
    ]

    last_retr_ms, last_gen_ms = 8.0, 18.0
    p95_budget_ms = 45.0

    for step, q in enumerate(traffic, 1):
        knobs = choose_knobs(p95_budget_ms, last_retr_ms, last_gen_ms)

        print(f"\n--- req {step} | k={knobs['top_k']} budget={knobs['token_budget']} "
              f"cheap={knobs['cheap_mode']} int8_store={knobs['use_int8_store']} ---")

        t0 = time.time()

        key = _fingerprint(q)
        cached = retr_cache.get(key)
        cache_hit = cached is not None

        # query embedding (fake, but stable)
        h = int(hashlib.md5(q.encode()).hexdigest(), 16) % (2**31 - 1)
        rs = cp.random.RandomState(h)
        qv = rs.standard_normal((1, doc_bank_fp16.shape[1]), dtype=cp.float16)
        qv /= cp.linalg.norm(qv, axis=1, keepdims=True) + 1e-6

        tr0 = time.time()
        if cache_hit:
            doc_ids = cached
        else:
            if knobs["use_int8_store"]:
                bank = int8_unpack(packed_i8, inv_scale)
            else:
                bank = doc_bank_fp16

            doc_ids = gpu_retrieve(qv, bank, knobs["top_k"])
            doc_ids = cp.asnumpy(doc_ids[0])
            retr_cache.put(key, doc_ids)
        cp.cuda.Stream.null.synchronize()
        tr1 = time.time()
        retr_ms = (tr1 - tr0) * 1000.0

        tc0 = time.time()
        chosen = fake_context_builder(doc_ids, knobs["token_budget"])
        tc1 = time.time()
        ctx_ms = (tc1 - tc0) * 1000.0

        tg0 = time.time()
        ans = fake_generate(q, chosen, knobs["cheap_mode"])
        tg1 = time.time()
        gen_ms = (tg1 - tg0) * 1000.0

        t1 = time.time()
        e2e_ms = (t1 - t0) * 1000.0

        print(ans)
        print(f"timing(ms): e2e={e2e_ms:.1f} retr={retr_ms:.1f} ctx={ctx_ms:.1f} gen={gen_ms:.1f} hit={'yes' if cache_hit else 'no'}")

        last_retr_ms, last_gen_ms = retr_ms, gen_ms


if __name__ == "__main__":
    main()