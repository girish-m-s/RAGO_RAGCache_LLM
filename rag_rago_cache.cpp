// rag_rago_cache.cpp
#include <algorithm>
#include <chrono>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;

static inline uint64_t now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             Clock::now().time_since_epoch())
      .count();
}

static inline void tiny_pause_ms(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

template <class K, class V>
class LruBox {
 public:
  explicit LruBox(size_t cap) : cap_(cap) {}

  std::optional<V> get(const K& key) {
    auto it = slots_.find(key);
    if (it == slots_.end()) return std::nullopt;

    touch(it->second.second);
    return it->second.first;
  }

  void put(const K& key, V value) {
    auto it = slots_.find(key);
    if (it != slots_.end()) {
      it->second.first = std::move(value);
      touch(it->second.second);
      return;
    }

    if (order_.size() >= cap_) {
      const K& old = order_.back();
      slots_.erase(old);
      order_.pop_back();
    }

    order_.push_front(key);
    slots_.emplace(key, std::make_pair(std::move(value), order_.begin()));
  }

  size_t size() const { return slots_.size(); }

 private:
  void touch(typename std::deque<K>::iterator it) {
    K key = *it;
    order_.erase(it);
    order_.push_front(std::move(key));
    slots_[order_.front()].second = order_.begin();
  }

  size_t cap_;
  std::deque<K> order_;
  std::unordered_map<K, std::pair<V, typename std::deque<K>::iterator>> slots_;
};

struct TuneKnobs {
  int top_k = 8;
  int batch = 8;
  bool cheap_mode = false; // pretend "quantized / smaller model"
};

struct Timings {
  double retrieval_ms = 0.0;
  double context_ms = 0.0;
  double gen_ms = 0.0;
  double e2e_ms = 0.0;
  bool cache_hit = false;
};

static std::vector<int> fake_retrieval(const std::string& q, int top_k) {
  // deterministic-ish "scores"
  std::seed_seq seed(q.begin(), q.end());
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> pick(0, 200000);

  std::vector<int> hits;
  hits.reserve(top_k);
  for (int i = 0; i < top_k; i++) hits.push_back(pick(rng));
  std::sort(hits.begin(), hits.end());
  hits.erase(std::unique(hits.begin(), hits.end()), hits.end());
  return hits;
}

static std::string fake_doc_text(int doc_id) {
  // just something stable
  return "Doc#" + std::to_string(doc_id) + " :: " +
         "A short block of evidence text used for grounding.";
}

static std::string build_context(const std::vector<int>& doc_ids,
                                 LruBox<int, std::string>& block_cache,
                                 int token_budget) {
  std::string stitched;
  stitched.reserve(static_cast<size_t>(token_budget) * 4);

  int tokens_left = token_budget;
  for (int id : doc_ids) {
    auto got = block_cache.get(id);
    std::string piece = got ? *got : fake_doc_text(id);
    if (!got) block_cache.put(id, piece);

    int cost = 40; // pretend tokens per block
    if (tokens_left - cost < 0) break;
    tokens_left -= cost;

    stitched += piece;
    stitched += "\n";
  }
  return stitched;
}

static std::string fake_generate(const std::string& question,
                                 const std::string& context,
                                 bool cheap_mode) {
  // generation cost grows with context length
  int base = cheap_mode ? 12 : 20;
  int extra = static_cast<int>(context.size() / 300);
  tiny_pause_ms(base + extra);

  return "Answer: " + question + "\n(grounded in " +
         std::to_string(context.size()) + " bytes of context)";
}

static TuneKnobs pick_knobs(double p95_budget_ms, double recent_retr_ms,
                            double recent_gen_ms) {
  TuneKnobs k;
  // crude “cost model”: if gen is expensive, reduce top_k and enable cheap_mode
  if (recent_gen_ms > p95_budget_ms * 0.55) {
    k.top_k = 6;
    k.cheap_mode = true;
  } else {
    k.top_k = 10;
    k.cheap_mode = false;
  }

  // if retrieval is bottleneck, batch more (pretend micro-batching)
  if (recent_retr_ms > p95_budget_ms * 0.25) k.batch = 16;
  else k.batch = 8;

  return k;
}

static Timings serve_one(const std::string& question,
                         LruBox<std::string, std::vector<int>>& retr_cache,
                         LruBox<int, std::string>& block_cache,
                         const TuneKnobs& knobs) {
  Timings t;
  uint64_t t0 = now_ms();

  // start retrieval async so we can “do stuff” in parallel (RAGCache-ish overlap)
  bool hit = false;
  std::vector<int> doc_ids;

  auto cached = retr_cache.get(question);
  if (cached) {
    hit = true;
    doc_ids = *cached;
  }

  std::future<std::vector<int>> retr_future;
  if (!hit) {
    retr_future = std::async(std::launch::async, [&]() {
      uint64_t a = now_ms();
      tiny_pause_ms(6); // pretend ANN time
      auto out = fake_retrieval(question, knobs.top_k);
      uint64_t b = now_ms();
      (void)a; (void)b;
      return out;
    });
  }

  // while retrieval runs, do a tiny “draft” step (toy)
  tiny_pause_ms(knobs.cheap_mode ? 2 : 3);

  uint64_t tr0 = now_ms();
  if (!hit) {
    doc_ids = retr_future.get();
    retr_cache.put(question, doc_ids);
  }
  uint64_t tr1 = now_ms();
  t.retrieval_ms = double(tr1 - tr0);
  t.cache_hit = hit;

  uint64_t tc0 = now_ms();
  int token_budget = knobs.cheap_mode ? 220 : 320;
  auto context = build_context(doc_ids, block_cache, token_budget);
  uint64_t tc1 = now_ms();
  t.context_ms = double(tc1 - tc0);

  uint64_t tg0 = now_ms();
  auto answer = fake_generate(question, context, knobs.cheap_mode);
  uint64_t tg1 = now_ms();
  t.gen_ms = double(tg1 - tg0);

  uint64_t t1 = now_ms();
  t.e2e_ms = double(t1 - t0);

  std::cout << answer << "\n";
  return t;
}

int main() {
  LruBox<std::string, std::vector<int>> retr_cache(512);
  LruBox<int, std::string> block_cache(4096);

  std::vector<std::string> traffic = {
      "what is rag latency?",
      "how to reduce rag cost?",
      "what is rag latency?",
      "explain caching in rag",
      "how to reduce rag cost?",
      "what is ragserve vs rago?"
  };

  double last_retr = 8.0, last_gen = 18.0;
  double p95_budget_ms = 40.0;

  for (int i = 0; i < (int)traffic.size(); i++) {
    auto knobs = pick_knobs(p95_budget_ms, last_retr, last_gen);

    std::cout << "\n--- request " << (i + 1)
              << " | top_k=" << knobs.top_k
              << " batch=" << knobs.batch
              << " cheap_mode=" << (knobs.cheap_mode ? "on" : "off")
              << " ---\n";

    auto t = serve_one(traffic[i], retr_cache, block_cache, knobs);

    last_retr = t.retrieval_ms;
    last_gen = t.gen_ms;

    std::cout << "timing(ms): e2e=" << t.e2e_ms
              << " retr=" << t.retrieval_ms
              << " ctx=" << t.context_ms
              << " gen=" << t.gen_ms
              << " cache_hit=" << (t.cache_hit ? "yes" : "no")
              << "\n";
  }

  std::cout << "\nretr_cache=" << retr_cache.size()
            << " block_cache=" << block_cache.size() << "\n";
  return 0;
}