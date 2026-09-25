#include <cmath>
#include <unordered_map>

#include "tilesight/engine.hpp"

namespace tilesight {
namespace {
double phi(double x) {  // Zelen & Severo (paper Eq. 10)
  if (x < 0) return 1.0 - phi(-x);
  const double t = 1.0 / (1.0 + 0.33267 * x);
  const double a1 = 0.4361836, a2 = -0.1201676, a3 = 0.9372980;
  return 1.0 - (a1 * t + a2 * t * t + a3 * t * t * t) * std::exp(-x * x / 2) / std::sqrt(2 * M_PI);
}

struct BIT {
  std::vector<int64_t> t;
  explicit BIT(size_t n) : t(n + 1, 0) {}
  void add(size_t i, int64_t v) { for (++i; i < t.size(); i += i & (~i + 1)) t[i] += v; }
  int64_t prefix(size_t i) const { int64_t s = 0; for (; i > 0; i -= i & (~i + 1)) s += t[i]; return s; }
};
}  // namespace

double hit_prob(int64_t d, int assoc, double cap) {
  if (d < assoc) return 1.0;
  if (cap <= assoc) return d >= cap ? 0.0 : 1.0;
  const double p = assoc / cap;
  if (d <= 256) {
    const double q = 1.0 - p;
    double term = std::pow(q, static_cast<double>(d)), acc = term;
    for (int a = 1; a < assoc; ++a) { term *= (static_cast<double>(d - a + 1) / a) * p / q; acc += term; }
    return std::min(1.0, acc);
  }
  const double mu = d * p, sigma = std::sqrt(d * p * (1 - p));
  return phi((assoc - 1 + 0.5 - mu) / sigma);
}

std::vector<double> expected_misses(const std::vector<int64_t>& keys, const std::vector<int>& streams,
                                    int n_streams, int assoc, double cap) {
  BIT bit(keys.size());
  std::unordered_map<int64_t, size_t> last;
  last.reserve(keys.size());
  std::vector<double> miss(n_streams, 0.0);
  for (size_t t = 0; t < keys.size(); ++t) {
    auto it = last.find(keys[t]);
    if (it != last.end()) {
      const size_t lt = it->second;
      const int64_t d = bit.prefix(t) - bit.prefix(lt + 1);
      miss[streams[t]] += 1.0 - hit_prob(d, assoc, cap);
      bit.add(lt, -1);
      it->second = t;
    } else {
      miss[streams[t]] += 1.0;
      last.emplace(keys[t], t);
    }
    bit.add(t, 1);
  }
  return miss;
}

}  // namespace tilesight
