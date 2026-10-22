// AssembleNIST: analytic ISO-metric thread SDFs and grid SDFs for MuJoCo (plugin library).
//
//   assemblenist.sdf.bolt : hex head (z in [0, head_top]) + externally threaded shank (z in [thread_z0, thread_z1])
//   assemblenist.sdf.nut  : hex prism (z in [z0, z1]) with an internal thread
//
// Thread profile: 60-degree V with flat crests/roots (trapezoid) in the axial half-plane (r, s),
// s = pitch * frac(z / pitch - theta / (2 pi) + phase), right-handed. The signed distance is the exact
// 2D distance to the periodic profile polyline (helix lead angle neglected, < 5 degrees here).
// Bolt material: r < r_s(s), r_s = clamp(rp + 0.433 P - tan60 |s - P/2|, root, crest).
// Nut material:  r > r_h(s), r_h = clamp(rp + c + 0.433 P - tan60 |s - P/2|, minor, major).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

#include <fstream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <immintrin.h>

#include <mujoco/mjplugin.h>
#include <mujoco/mujoco.h>

#ifndef ASSEMBLENIST_SDF_DIR
#define ASSEMBLENIST_SDF_DIR "."
#endif

namespace {

constexpr double kTan60 = 1.7320508075688772;
constexpr double kPi = 3.14159265358979323846;

inline double frac(double x) { return x - std::floor(x); }

// Hexagon (flats at distance a from the axis, one flat normal along +x) 2D signed distance.
double hexagon2d(double x, double y, double a) {
  const double kx = -0.8660254037844386, ky = 0.5, kz = 0.5773502691896258;
  x = std::fabs(x); y = std::fabs(y);
  // rotate so that flats are perpendicular to x
  double px = y, py = x;
  double d = 2.0 * std::min(kx * px + ky * py, 0.0);
  px -= d * kx; py -= d * ky;
  double cx = std::clamp(px, -kz * a, kz * a);
  double qx = px - cx, qy = py - a;
  double len = std::sqrt(qx * qx + qy * qy);
  return (qy < 0 ? -len : len);
}

double extrude(double d2, double z, double z0, double z1) {
  double hz = 0.5 * (z1 - z0), cz = 0.5 * (z1 + z0);
  double wx = d2, wy = std::fabs(z - cz) - hz;
  double out = std::sqrt(std::max(wx, 0.0) * std::max(wx, 0.0) + std::max(wy, 0.0) * std::max(wy, 0.0));
  return std::min(std::max(wx, wy), 0.0) + out;
}

// distance from (r, s) to segment (a, b); returns distance and the closest-point offset (dr, ds)
inline double segdist(double r, double s, double ar, double as, double br, double bs, double* dr, double* ds) {
  double vr = br - ar, vs = bs - as;
  double t = ((r - ar) * vr + (s - as) * vs) / (vr * vr + vs * vs);
  t = t < 0 ? 0 : (t > 1 ? 1 : t);
  *dr = r - (ar + t * vr); *ds = s - (as + t * vs);
  return std::sqrt(*dr * *dr + *ds * *ds);
}

// Signed distance to a periodic trapezoid profile r_p(s) = clamp(top - tan60 |s - P/2|, lo, hi), s in [0, P).
// below=true: material where r < r_p(s). Also returns d(dist)/dr and d(dist)/ds.
double thread2d(double r, double s, double P, double top, double lo, double hi, bool below, double* gr, double* gs) {
  double s1 = 0.5 * P - (top - hi) / kTan60;  // crest start
  double s0 = 0.5 * P - (top - lo) / kTan60;  // root end
  s0 = std::max(s0, 0.0); s1 = std::max(s1, s0);
  double s2 = P - s1, s3 = P - s0;
  // profile polyline over [-P + s3, P + s0]: previous falling flank, this period, next rising flank
  const double pts[8][2] = {{hi, s2 - P}, {lo, s3 - P}, {lo, s0}, {hi, s1}, {hi, s2}, {lo, s3}, {lo, P + s0}, {hi, P + s1}};
  double best = 1e30, br = 0, bs = 0;
  for (int i = 0; i < 7; ++i) {
    double dr, ds;
    double d = segdist(r, s, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], &dr, &ds);
    if (d < best) { best = d; br = dr; bs = ds; }
  }
  double rs = std::clamp(top - kTan60 * std::fabs(s - 0.5 * P), lo, hi);
  bool inside = below ? (r < rs) : (r > rs);
  double sgn = inside ? -1.0 : 1.0;
  if (best > 1e-12) { *gr = sgn * br / best; *gs = sgn * bs / best; }
  else { *gr = below ? 1.0 : -1.0; *gs = 0.0; }
  return sgn * best;
}

// ---------------------------------------------------------------------------------------------
// attribute order (bolt): pitch, rp, root, crest, thread_z0, thread_z1, head_flat, head_top, phase
const char* kBoltNames[] = {"pitch", "rp", "root", "crest", "thread_z0", "thread_z1", "head_flat", "head_top", "phase"};
const double kBoltDefaults[] = {0.002, 0.00719, 0.00639, 0.00784, 0.01, 0.035, 0.012, 0.01, 0.0};
constexpr int kBoltN = 9;
// attribute order (nut): pitch, rp, clearance, minor, major, flat, z0, z1, phase
const char* kNutNames[] = {"pitch", "rp", "clearance", "minor", "major", "flat", "z0", "z1", "phase"};
const double kNutDefaults[] = {0.002, 0.00719, 0.0001, 0.0069, 0.00795, 0.012, 0.01, 0.023, 0.0};
constexpr int kNutN = 9;

// helix coordinates: s = P frac(z/P - theta/2pi + phase); ds/dx, ds/dy, ds/dz
inline double helix(const mjtNum p[3], double P, double ph, double* r, double J[3]) {
  double rr2 = p[0] * p[0] + p[1] * p[1];
  *r = std::sqrt(rr2);
  double th = std::atan2(p[1], p[0]);
  double s = P * frac(p[2] / P - th / (2 * kPi) + ph);
  double k = -P / (2 * kPi) / std::max(rr2, 1e-18);   // d(-P theta/2pi)/dx = P y/(2pi r^2) ...
  J[0] = -k * p[1]; J[1] = k * p[0]; J[2] = 1.0;
  return s;
}

// Bolt: externally threaded shank only (the head is embedded below the board surface).
double bolt_eval(const mjtNum p[3], const mjtNum* a, mjtNum* g) {
  double P = a[0], rp = a[1], root = a[2], crest = a[3], tz0 = a[4], tz1 = a[5], ph = a[8];
  double r, J[3];
  double s = helix(p, P, ph, &r, J);
  double gr, gs;
  double d = thread2d(r, s, P, rp + 0.25 * P * kTan60, root, crest, true, &gr, &gs);
  double gx = gr * (r > 1e-12 ? p[0] / r : 0) + gs * J[0];
  double gy = gr * (r > 1e-12 ? p[1] / r : 0) + gs * J[1];
  double gz = gs * J[2];
  // caps: below the thread start and above the top (with a 45-degree chamfer)
  double cap0 = tz0 - p[2], cap1 = p[2] - tz1;
  double cham = (p[2] - tz1 + (r - root)) * 0.7071067811865476;
  if (cap0 > d) { d = cap0; gx = 0; gy = 0; gz = -1; }
  if (cap1 > d) { d = cap1; gx = 0; gy = 0; gz = 1; }
  if (cham > d) { d = cham; gx = 0.7071067811865476 * (r > 1e-12 ? p[0] / r : 0);
                  gy = 0.7071067811865476 * (r > 1e-12 ? p[1] / r : 0); gz = 0.7071067811865476; }
  if (g) { g[0] = gx; g[1] = gy; g[2] = gz; }
  return d;
}

double nut_eval(const mjtNum p[3], const mjtNum* a, mjtNum* g) {
  double P = a[0], rp = a[1], c = a[2], minor = a[3], major = a[4], flat = a[5], z0 = a[6], z1 = a[7], ph = a[8];
  double r, J[3];
  double s = helix(p, P, ph, &r, J);
  double gr, gs;
  double hole = thread2d(r, s, P, rp + c + 0.25 * P * kTan60, minor, major, false, &gr, &gs);
  double body = extrude(hexagon2d(p[0], p[1], flat), p[2], z0, z1);
  if (body > hole) {
    if (g) {  // finite-difference gradient for the (rarely active) outer hex body
      const double eps = 1e-7;
      for (int i = 0; i < 3; ++i) {
        mjtNum q[3] = {p[0], p[1], p[2]}; q[i] += eps;
        g[i] = (extrude(hexagon2d(q[0], q[1], flat), q[2], z0, z1) - body) / eps;
      }
    }
    return body;
  }
  if (g) {
    g[0] = gr * (r > 1e-12 ? p[0] / r : 0) + gs * J[0];
    g[1] = gr * (r > 1e-12 ? p[1] / r : 0) + gs * J[1];
    g[2] = gs * J[2];
  }
  return hole;
}

double bolt_distance(const mjtNum p[3], const mjtNum* a) { return bolt_eval(p, a, nullptr); }
double nut_distance(const mjtNum p[3], const mjtNum* a) { return nut_eval(p, a, nullptr); }


// ---------------------------------------------------------------------------------------------
// Grid SDF (connectors): float16 samples on a regular grid, trilinear interpolation.
struct Grid {
  int nx = 0, ny = 0, nz = 0;
  double ox = 0, oy = 0, oz = 0, h = 1;
  std::vector<uint16_t> v;
};
std::mutex g_mu;
std::unordered_map<int, std::unique_ptr<Grid>> g_grids;

const Grid* get_grid(int id) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_grids.find(id);
  if (it != g_grids.end()) return it->second.get();
  const char* env = std::getenv("ASSEMBLENIST_SDF_DIR");
  std::string path = std::string(env && *env ? env : ASSEMBLENIST_SDF_DIR) + "/grid_" + std::to_string(id) + ".bin";
  std::ifstream f(path, std::ios::binary);
  auto g = std::make_unique<Grid>();
  if (!f) { mju_warning("assemblenist.sdf.grid: cannot open %s", path.c_str()); g->nx = g->ny = g->nz = 2; g->v.assign(8, 0x5c00); }
  else {
    int n[3]; double o[4];
    f.read(reinterpret_cast<char*>(n), sizeof(n)); f.read(reinterpret_cast<char*>(o), sizeof(o));
    g->nx = n[0]; g->ny = n[1]; g->nz = n[2]; g->ox = o[0]; g->oy = o[1]; g->oz = o[2]; g->h = o[3];
    g->v.resize(size_t(n[0]) * n[1] * n[2]);
    f.read(reinterpret_cast<char*>(g->v.data()), g->v.size() * sizeof(uint16_t));
  }
  const Grid* ptr = g.get();
  g_grids[id] = std::move(g);
  return ptr;
}

inline float h2f(uint16_t x) { return _cvtsh_ss(x); }

double grid_eval(const mjtNum p[3], const Grid* G, mjtNum* grad) {
  const double h = G->h;
  double f[3] = {(p[0] - G->ox) / h, (p[1] - G->oy) / h, (p[2] - G->oz) / h};
  const int n[3] = {G->nx, G->ny, G->nz};
  double c[3], e[3];
  int i0[3];
  double t[3];
  for (int k = 0; k < 3; ++k) {
    c[k] = std::clamp(f[k], 0.0, n[k] - 1.000001);
    e[k] = (f[k] - c[k]) * h;                 // offset outside the grid box (metres)
    i0[k] = int(std::floor(c[k]));
    t[k] = c[k] - i0[k];
  }
  auto at = [&](int x, int y, int z) { return double(h2f(G->v[(size_t(z) * n[1] + y) * n[0] + x])); };
  double c000 = at(i0[0], i0[1], i0[2]), c100 = at(i0[0] + 1, i0[1], i0[2]);
  double c010 = at(i0[0], i0[1] + 1, i0[2]), c110 = at(i0[0] + 1, i0[1] + 1, i0[2]);
  double c001 = at(i0[0], i0[1], i0[2] + 1), c101 = at(i0[0] + 1, i0[1], i0[2] + 1);
  double c011 = at(i0[0], i0[1] + 1, i0[2] + 1), c111 = at(i0[0] + 1, i0[1] + 1, i0[2] + 1);
  double tx = t[0], ty = t[1], tz = t[2];
  double c00 = c000 + (c100 - c000) * tx, c10 = c010 + (c110 - c010) * tx;
  double c01 = c001 + (c101 - c001) * tx, c11 = c011 + (c111 - c011) * tx;
  double c0 = c00 + (c10 - c00) * ty, c1 = c01 + (c11 - c01) * ty;
  double d = c0 + (c1 - c0) * tz;
  double out = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  if (grad) {
    if (out > 1e-12) {
      for (int k = 0; k < 3; ++k) grad[k] = e[k] / out;
    } else {
      double gx = ((c100 - c000) * (1 - ty) + (c110 - c010) * ty) * (1 - tz) + ((c101 - c001) * (1 - ty) + (c111 - c011) * ty) * tz;
      double gy = (c10 - c00) * (1 - tz) + (c11 - c01) * tz;
      double gz = c1 - c0;
      grad[0] = gx / h; grad[1] = gy / h; grad[2] = gz / h;
    }
  }
  return d + out;
}
const char* kGridNames[] = {"id"};
const double kGridDefaults[] = {0};

template <int N>
struct Attr {
  static void Read(const mjModel* m, int instance, const char* const* names, const double* defaults, mjtNum* out) {
    for (int i = 0; i < N; ++i) {
      const char* v = mj_getPluginConfig(m, instance, names[i]);
      out[i] = (v && *v) ? std::strtod(v, nullptr) : defaults[i];
    }
  }
  static void FromStrings(mjtNum* out, const char* const* names, const double* defaults, const char* name[], const char* value[]) {
    for (int i = 0; i < N; ++i) {
      out[i] = defaults[i];
      for (int j = 0; j < N; ++j)
        if (name[j] && std::strcmp(name[j], names[i]) == 0 && value[j] && *value[j]) out[i] = std::strtod(value[j], nullptr);
    }
  }
};

struct Inst { mjtNum a[16]; };

template <int N, double (*DIST)(const mjtNum*, const mjtNum*)>
void gradient(mjtNum g[3], const mjtNum p[3], const mjtNum* a) {
  const double eps = 1e-7;
  double d0 = DIST(p, a);
  for (int i = 0; i < 3; ++i) {
    mjtNum q[3] = {p[0], p[1], p[2]};
    q[i] += eps;
    g[i] = (DIST(q, a) - d0) / eps;
  }
}

void register_one(const char* name, int n, const char* const* names, const double* defaults,
                  double (*dist)(const mjtNum*, const mjtNum*), void (*aabb)(mjtNum[6], const mjtNum*),
                  void (*grad)(mjtNum[3], const mjtNum[3], const mjtNum*)) {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = name;
  plugin.capabilityflags |= mjPLUGIN_SDF;
  plugin.nattribute = n;
  plugin.attributes = names;
  plugin.nstate = +[](const mjModel*, int) { return 0; };
  static std::map<std::string, std::pair<const char* const*, const double*>> table;
  table[name] = {names, defaults};
  // init/destroy store per-instance attributes in plugin_data
  if (std::strcmp(name, "assemblenist.sdf.bolt") == 0) {
    plugin.init = +[](const mjModel* m, mjData* d, int instance) {
      auto* in = new Inst();
      Attr<kBoltN>::Read(m, instance, kBoltNames, kBoltDefaults, in->a);
      d->plugin_data[instance] = reinterpret_cast<uintptr_t>(in);
      return 0;
    };
    plugin.sdf_distance = +[](const mjtNum p[3], const mjData* d, int instance) {
      return bolt_distance(p, reinterpret_cast<Inst*>(d->plugin_data[instance])->a);
    };
    plugin.sdf_gradient = +[](mjtNum g[3], const mjtNum p[3], const mjData* d, int instance) {
      bolt_eval(p, reinterpret_cast<Inst*>(d->plugin_data[instance])->a, g);
    };
    plugin.sdf_staticdistance = +[](const mjtNum p[3], const mjtNum* a) { return bolt_distance(p, a); };
    plugin.sdf_attribute = +[](mjtNum a[], const char* nm[], const char* val[]) {
      Attr<kBoltN>::FromStrings(a, kBoltNames, kBoltDefaults, nm, val);
    };
    plugin.sdf_aabb = +[](mjtNum b[6], const mjtNum* a) {
      double R = a[3] + 0.0003;  // threaded shank only
      b[0] = 0; b[1] = 0; b[2] = 0.5 * (a[4] + a[5]); b[3] = R; b[4] = R; b[5] = 0.5 * (a[5] - a[4]) + 0.0003;
    };
  } else if (std::strcmp(name, "assemblenist.sdf.grid") == 0) {
    plugin.init = +[](const mjModel* m, mjData* d, int instance) {
      auto* in = new Inst();
      Attr<1>::Read(m, instance, kGridNames, kGridDefaults, in->a);
      in->a[1] = 0;
      d->plugin_data[instance] = reinterpret_cast<uintptr_t>(in);
      get_grid(int(in->a[0]));
      return 0;
    };
    plugin.sdf_distance = +[](const mjtNum p[3], const mjData* d, int instance) {
      return grid_eval(p, get_grid(int(reinterpret_cast<Inst*>(d->plugin_data[instance])->a[0])), nullptr);
    };
    plugin.sdf_gradient = +[](mjtNum g[3], const mjtNum p[3], const mjData* d, int instance) {
      grid_eval(p, get_grid(int(reinterpret_cast<Inst*>(d->plugin_data[instance])->a[0])), g);
    };
    plugin.sdf_staticdistance = +[](const mjtNum p[3], const mjtNum* a) { return grid_eval(p, get_grid(int(a[0])), nullptr); };
    plugin.sdf_attribute = +[](mjtNum a[], const char* nm[], const char* val[]) {
      Attr<1>::FromStrings(a, kGridNames, kGridDefaults, nm, val);
    };
    plugin.sdf_aabb = +[](mjtNum b[6], const mjtNum* a) {
      const Grid* G = get_grid(int(a[0]));
      double hx = 0.5 * (G->nx - 1) * G->h, hy = 0.5 * (G->ny - 1) * G->h, hz = 0.5 * (G->nz - 1) * G->h;
      b[0] = G->ox + hx; b[1] = G->oy + hy; b[2] = G->oz + hz; b[3] = hx; b[4] = hy; b[5] = hz;
    };
  } else {
    plugin.init = +[](const mjModel* m, mjData* d, int instance) {
      auto* in = new Inst();
      Attr<kNutN>::Read(m, instance, kNutNames, kNutDefaults, in->a);
      d->plugin_data[instance] = reinterpret_cast<uintptr_t>(in);
      return 0;
    };
    plugin.sdf_distance = +[](const mjtNum p[3], const mjData* d, int instance) {
      return nut_distance(p, reinterpret_cast<Inst*>(d->plugin_data[instance])->a);
    };
    plugin.sdf_gradient = +[](mjtNum g[3], const mjtNum p[3], const mjData* d, int instance) {
      nut_eval(p, reinterpret_cast<Inst*>(d->plugin_data[instance])->a, g);
    };
    plugin.sdf_staticdistance = +[](const mjtNum p[3], const mjtNum* a) { return nut_distance(p, a); };
    plugin.sdf_attribute = +[](mjtNum a[], const char* nm[], const char* val[]) {
      Attr<kNutN>::FromStrings(a, kNutNames, kNutDefaults, nm, val);
    };
    plugin.sdf_aabb = +[](mjtNum b[6], const mjtNum* a) {
      double R = a[5] * 1.1547005383792515 + 0.0005;
      b[0] = 0; b[1] = 0; b[2] = 0.5 * (a[6] + a[7]); b[3] = R; b[4] = R; b[5] = 0.5 * (a[7] - a[6]) + 0.0005;
    };
  }
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<Inst*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel*, mjtNum*, void*, int) {};
  plugin.compute = +[](const mjModel*, mjData*, int, int) {};
  mjp_registerPlugin(&plugin);
}

}  // namespace

mjPLUGIN_LIB_INIT(assemblenist_sdf) {
  register_one("assemblenist.sdf.bolt", kBoltN, kBoltNames, kBoltDefaults, bolt_distance, nullptr, nullptr);
  register_one("assemblenist.sdf.nut", kNutN, kNutNames, kNutDefaults, nut_distance, nullptr, nullptr);
  register_one("assemblenist.sdf.grid", 1, kGridNames, kGridDefaults, nullptr, nullptr, nullptr);
}
