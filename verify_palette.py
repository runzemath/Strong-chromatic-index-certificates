"""Check the interval bounds and integer recurrence for Appendix C."""

from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import isqrt
from multiprocessing import get_context
from pathlib import Path


M = 100_000_000
M2 = M * M
M3 = M2 * M
SCALE = 1_000_000_000_000
BASE = 66_521_250
REFERENCE = 86_485_000
K_LO, K_HI = 185_480_000, 185_481_000
TRIANGLE_MAX, STAR_MAX = 61_700_000, 61_800_000
TRIANGLE_MULTIPLIER = 10_000_000
TRIANGLE_SLOPE = 35_100_000
MOMENTS = (
    (18_000_000, 8_000_000, 6_410_000, 28_100_000, 6_140_000, 29_000_000),
    (24_000_000, 2_500_000, 6_865_000, 26_800_000, 6_321_000, 28_500_000),
)
TAIL = (
    (77_000_000, 30_026_000, 0),
    (70_000_000, 8_851_000, 27_500_000),
    (62_000_000, 6_681_000, 30_600_000),
    (53_000_000, 5_379_000, 32_700_000),
    (43_000_000, 2_835_000, 37_500_000),
)
SHARPER_TAIL = (
    (77_000_000, 30_151_000, 0),
    (70_000_000, 8_976_000, 27_500_000),
    (66_000_000, 7_156_000, 30_100_000),
    (60_000_000, 6_364_000, 31_300_000),
    (54_000_000, 5_644_000, 32_500_000),
    (50_000_000, 3_700_000, 36_100_000),
    (43_000_000, 2_750_000, 38_000_000),
)


def ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class Schedule:
    q: int
    cutoff: int
    claim: int
    cuts: tuple
    density: tuple
    bonuses: tuple


class Envelope:
    """Locate minima of x*r + y over the points (x, y) in one family."""

    def __init__(self, lines):
        by_slope = {}
        for x, y, q in lines:
            slope = Fraction(x, q)
            if slope in by_slope:
                _, previous_y, previous_q = by_slope[slope]
                assert y * previous_q == previous_y * q
            by_slope[slope] = (x, y, q)
        self.lines = tuple(line for _, line in sorted(by_slope.items()))
        breaks = []
        for (x, y, q), (xx, yy, qq) in zip(self.lines, self.lines[1:]):
            breaks.append(Fraction(yy * q - y * qq, xx * q - x * qq))
        assert all(a <= b for a, b in zip(breaks, breaks[1:]))
        self.integral = all(b.denominator == 1 for b in breaks)
        self.breaks = tuple(int(b) if self.integral else b for b in breaks)

    def minimum_index(self, numerator, denominator=1):
        r = ceil_div(numerator, denominator) if self.integral else Fraction(numerator, denominator)
        return bisect_right(self.breaks, -r)


def reference_lines(weights, cuts):
    # Supporting lines used in verify_palette.cpp. Store x = -p/q and
    # y = (a+p*M)/q as the triples (-p, a+p*M, q).
    hull = []
    for w, d in zip(weights, cuts):
        if hull and hull[-1][0] == w:
            continue
        while len(hull) > 1:
            w0, d0 = hull[-2]
            w1, d1 = hull[-1]
            if (d1 - d0) * (w - w1) < (d - d1) * (w1 - w0):
                break
            hull.pop()
        hull.append((w, d))
    slopes = [(0, 1)]
    slopes.extend((d1 - d0, w1 - w0)
                  for (w0, d0), (w1, d1) in zip(hull, hull[1:]))
    cross_x, cross_q = 0, 1
    for w, d in zip(weights, cuts):
        if d < REFERENCE and w < M and (REFERENCE - d) * cross_q > cross_x * (M - w):
            cross_x, cross_q = REFERENCE - d, M - w
    if cross_x:
        slopes.append((cross_x, cross_q))
    points = tuple((M - w, REFERENCE - d) for w, d in zip(weights, cuts))
    return [(x, max(q * b - x * v for v, b in points), q) for x, q in slopes]


@lru_cache(maxsize=48_000)
def degree_data(dl, dh, density):
    pairs = {(BASE, BASE)}
    for intercept, slope in density:
        pairs.add((intercept - slope + ceil_div(slope * dl, M),
                   intercept - slope + ceil_div(slope * dh, M)))
    # A pair greater at both endpoints can never lower the budget.
    frontier = []
    smallest_hi = M + 1
    for lo, hi in sorted(pairs):
        if hi < smallest_hi:
            frontier.append((lo, hi))
            smallest_hi = hi
    return ceil_div(dh * dh, M), tuple(frontier)


def linear_gain(s, pieces):
    for start, intercept, slope in pieces:
        if s >= start:
            return intercept + slope * s // M
    return 0


class Phase:
    def __init__(self, schedule, weights):
        self.smallest = weights[0]
        self.width = M - self.smallest
        self.smallest_squared = ceil_div(self.smallest**2, M)
        self.threshold = schedule.cuts[0]
        self.density = schedule.density
        self.bonuses = schedule.bonuses
        self.bonus_starts = tuple(start for start, _ in self.bonuses)
        lines = reference_lines(weights, schedule.cuts)
        self.nonnegative = Envelope(line for line in lines if line[1] >= 0)
        self.negative = Envelope(line for line in lines if line[1] < 0)
        slope = ceil_div(K_HI * (REFERENCE - self.threshold), M)
        cap, cap_slope = M, M
        for intercept, gradient in self.density:
            candidate = intercept - gradient + ceil_div((gradient + slope) * self.threshold, M)
            if candidate < cap:
                cap, cap_slope = candidate, gradient + slope
        self.triangle_caps = [(cap, cap_slope)]
        beta = max(ceil_div(M * (w - self.smallest), d - self.threshold)
                   for w, d in zip(weights[1:], schedule.cuts[1:]))
        factor = ceil_div((M + self.smallest) * beta, M)
        if factor * M2 <= K_LO * self.smallest**2:
            cap, slope = self.triangle_caps[0]
            cap = (ceil_div(self.smallest**2 * cap, M2)
                   + ceil_div(factor * (M - REFERENCE) * self.threshold * (M - self.threshold), M3))
            self.triangle_caps.append((cap, ceil_div(self.smallest**2 * slope, M2)))
        self.boxes = 0
        self.budget_cache = {}

    def size_bound(self, dh, uh):
        mean_hi = self.smallest + (self.width * uh + M - 1) // M
        return (dh * mean_hi + M - 1) // M

    def budget(self, dl, dh, ul):
        key = (dl, dh, ul)
        value = self.budget_cache.get(key)
        if value is not None:
            return value
        budget, density = degree_data(dl, dh, self.density)
        z = self.width - self.width * ul // M
        pos, neg = self.nonnegative, self.negative
        if pos.lines:
            x, y, q = pos.lines[pos.minimum_index(z)]
            den = q * M2
            raw = K_HI * (x * z + y)
            low, high = (raw * dl + den - 1) // den, (raw * dh + den - 1) // den
            for a, b in density:
                budget = min(budget, max(a + low, b + high))
        if neg.lines:
            # For y < 0, raw/q = (2*d-M)*(x*r + y),
            # where r = d*z/(2*d-M), with x and y written as fractions.
            # For a degree endpoint, the values decrease then increase
            # along the envelope. For the maximum of two endpoints, only
            # the lines between their two minimizing indices can improve
            # the bound. Rounding up preserves this monotonicity.
            lo = neg.minimum_index(dl * z, 2 * dl - M)
            hi = neg.minimum_index(dh * z, 2 * dh - M)
            for i in range(lo, hi + 1):
                x, y, q = neg.lines[i]
                den = q * M2
                raw_low = dl * x * z + (2 * dl - M) * y
                raw_high = dh * x * z + (2 * dh - M) * y
                low = ((K_HI if raw_low >= 0 else K_LO) * raw_low + den - 1) // den
                high = ((K_HI if raw_high >= 0 else K_LO) * raw_high + den - 1) // den
                for a, b in density:
                    budget = min(budget, max(a + low, b + high))
        assert budget >= 0
        self.budget_cache[key] = budget
        return budget

    def profile_density(self, dl, dh, ul, uh):
        budget = self.budget(dl, dh, ul)
        high_mass = (dh * uh + M - 1) // M
        hh = min(budget, (high_mass * high_mass + M - 1) // M)
        um = max(ul, min(M // 2, uh))
        mixed = (2 * dh * dh * um * (M - um) + M3 - 1) // M3
        cross = min(budget - hh, mixed)
        reference = (hh * M + cross * self.smallest
                     + (budget - hh - cross) * self.smallest_squared + M - 1) // M
        return budget, reference

    def degree_sum_cap(self, dl, dh):
        def gain(d):
            return ceil_div(K_HI * ((REFERENCE - self.threshold) * d
                                   + (M - REFERENCE) * (M - d)), M2)
        lo, hi = gain(dl), gain(dh)
        _, density = degree_data(dl, dh, self.density)
        return min(max(lo + a, hi + b) for a, b in density)

    def triangle_ok(self, dl, dh, ul, uh, cap, slope):
        limit = cap + slope * (dl - self.threshold) // M
        size = self.size_bound(dh, uh)
        if size * size <= limit * M:
            return True
        return self.profile_density(dl, dh, ul, uh)[1] <= limit

    def hit_ok(self, dl, dh, ul, uh, cap):
        t = self.size_bound(dh, uh)
        if t <= cap:
            return True
        budget, c_ref = self.profile_density(dl, dh, ul, uh)
        c = min(c_ref, (t * t + M - 1) // M)
        s0 = max(0, t * t // M - c)
        if t - s0 * (3 * M - isqrt(M * s0) - 1) // (6 * M) <= cap:
            return True
        x = M - t
        exponential = (M + x + x*x // (2*M) + x**3 // (6*M2)
                       + x**4 // (24*M3))
        required = ceil_div((M - cap) * M, exponential)
        s = M - c
        assert s >= self.bonus_starts[0]
        gain = 1_800_000 + 38_500_000 * min(s, 44_600_000) // M
        i = bisect_right(self.bonus_starts, s) - 1
        start, value = self.bonuses[i]
        if i and s - start < 10_000:
            previous = self.bonuses[i - 1][1]
            gain += previous + (value - previous) * (s - start) // 10_000
        else:
            gain += value
        if (gain >= required or s * (3*M - isqrt(M*s) - 1) // (6*M) >= required
                or linear_gain(s, TAIL) >= required or linear_gain(s, SHARPER_TAIL) >= required):
            return True

        s_ref = M - c_ref
        if not 25_000_000 <= s_ref <= 48_000_000 or required > 20_835_000:
            return False
        difference = max(0, self.degree_sum_cap(dl, dh) - budget)
        smallest_sq = self.smallest**2
        for bound, slope in self.triangle_caps:
            if s_ref >= 25_500_000:
                gap, increase = max(0, TRIANGLE_MAX - bound), max(0, bound - TRIANGLE_MAX)
                derivative = ceil_div(TRIANGLE_MULTIPLIER * gap * max(0, 4*c_ref - M), M2)
                if ((TRIANGLE_SLOPE - derivative) * K_LO * smallest_sq
                        >= TRIANGLE_MULTIPLIER * slope * c_ref * M):
                    correction = (ceil_div(TRIANGLE_MULTIPLIER * slope * c_ref * difference, M2*K_LO)
                                  + ceil_div(TRIANGLE_MULTIPLIER * increase, M))
                    bonus = TRIANGLE_MULTIPLIER * gap * c_ref * (2*c_ref - M) // M3
                    if 3_775_000 + TRIANGLE_SLOPE * s_ref // M - correction + bonus >= required:
                        return True
            if s_ref > 42_000_000 or required > 18_711_000:
                continue
            dgap, increase = max(0, STAR_MAX - bound), max(0, bound - STAR_MAX)
            root = isqrt(M * c_ref) + 1
            c_cube = c_ref**3 // M2
            for j, (star, clique, b_lo, a_lo, b_hi, a_hi) in enumerate(MOMENTS):
                breakpoint = 30_000_000 if j == 0 else 32_000_000
                line = b_lo + a_lo * s_ref // M if s_ref <= breakpoint else b_hi + a_hi * s_ref // M
                remaining_gap, slope_min = dgap, a_lo
                if j == 1 and dgap:
                    low_line = (7_600_000 + 26_800_000 * s_ref // M if s_ref < 32_000_000
                                else 7_760_000 + 26_300_000 * s_ref // M)
                    assert low_line >= line
                    portion = min(6_000_000, dgap)
                    line += (low_line - line) * portion // 6_000_000
                    remaining_gap -= portion
                    slope_min = 26_300_000
                factor = ceil_div(star * t**3, M3) + ceil_div(clique * c_ref * root, M2)
                derivative = ceil_div(3 * star * remaining_gap * c_ref * c_ref, M3)
                if (slope_min - derivative) * K_LO * smallest_sq < slope * factor * M2:
                    continue
                correction = (ceil_div(slope * factor * difference, M*K_LO)
                              + ceil_div((star + clique) * increase, M))
                addition = star * remaining_gap * c_cube // M2
                if line + addition - correction >= required:
                    return True
        return False

    def check_interval(self, lower, upper, check, *bound):
        stack = [(lower, upper, 0, M if self.width else 0)]
        width = self.width
        boxes = 0
        while stack:
            dl, dh, ul, uh = stack.pop()
            boxes += 1
            if check(dl, dh, ul, uh, *bound):
                continue
            if max(dh - dl, uh - ul) <= 1:
                raise ValueError(f"Unverified interval bound {bound} at {(dl, dh, ul, uh)}")
            if dh - dl > 1 and (uh - ul <= 1 or dh - dl >= (uh - ul) * width // M):
                d = (dl + dh) // 2
                stack.append((dl, d, ul, uh))
                stack.append((d, dh, ul, uh))
            else:
                u = (ul + uh) // 2
                assert ul < u < uh
                stack.append((dl, dh, ul, u))
                stack.append((dl, dh, u, uh))
        self.boxes += boxes


def check_phase(job):
    number, schedule, weights, triangle, caps = job
    phase = Phase(schedule, weights)
    try:
        for cap, slope in zip(triangle, (74_000_000, 98_000_000)):
            phase.check_interval(schedule.cuts[0], M, phase.triangle_ok, cap, slope)
            phase.triangle_caps.append((cap, slope))
        for lo, hi, cap in zip(schedule.cuts, schedule.cuts[1:] + (M,), caps):
            phase.check_interval(lo, hi, phase.hit_ok, cap)
    except ValueError as err:
        raise ValueError(f"Palette phase {number}: {err}") from err
    return phase.boxes


def phase_step(q, scale):
    if scale < SCALE // 50:
        return min(q, 45)
    if scale < SCALE // 10:
        return min(q, 85)
    if scale < 2 * SCALE // 5:
        return min(q, 185)
    return q


def prepare_checks(schedule, bounds):
    assert M // 2 < schedule.cuts[0] <= REFERENCE
    assert BASE + ceil_div(K_HI * (REFERENCE - schedule.cuts[0]), M) < M - schedule.bonuses[0][0]
    assert all(s + 10_000 < t and b <= c
               for (s, b), (t, c) in zip(schedule.bonuses, schedule.bonuses[1:]))
    assert all(0 < intercept <= M and slope >= 0 for intercept, slope in schedule.density)
    weights = (M,) * len(schedule.cuts)
    scale, palette = SCALE, 0
    checks = []
    rows = iter(bounds)
    while scale > schedule.cutoff:
        scans, caps = read_row(next(rows), len(weights))
        assert all(a <= b for a, b in zip(caps, caps[1:]))
        assert weights[-1] == M and all(a <= b for a, b in zip(weights, weights[1:]))
        assert all(0 < w <= M for w in weights)
        checks.append((len(checks) + 1, schedule, weights, scans, caps))
        q, peak = phase_step(schedule.q, scale), caps[-1]
        palette += ceil_div(scale * peak, M * q)
        updated = []
        for w, cap in zip(weights, caps):
            assert 0 < cap <= peak < 2 * cap
            r = peak - cap
            denominator = 2 * q*q * cap*cap
            numerator = denominator - 2*q*cap*r - r*(cap - r)
            updated.append(ceil_div(w * numerator, denominator))
        weights = tuple(updated)
        smaller = ceil_div(scale * (q - 1), q)
        assert smaller < scale
        scale = smaller
    assert weights[-1] == M and all(0 < w <= M for w in weights)
    assert all(a <= b for a, b in zip(weights, weights[1:]))
    scans, caps = read_row(next(rows), len(weights))
    checks.append((len(checks) + 1, schedule, weights, scans, caps))
    assert next(rows, None) is None
    return checks, palette + ceil_div(scale * max(caps), M)


def read_row(line, classes):
    values = tuple(map(int, line.split()))
    assert len(values) == classes + 2
    assert all(0 < x <= M for x in values)
    return values[:2], values[2:]


def verify_palette(schedule, parameters, jobs):
    path = Path(__file__).resolve().with_name("palette_bounds.txt")
    bounds = path.read_text(encoding="ascii").splitlines()
    checks, total = prepare_checks(schedule, bounds)
    phases = len(checks) - 1
    assert total == parameters["cap"] and phases == parameters["phases"]
    assert total < schedule.claim and schedule.claim * M <= schedule.cuts[0] * SCALE

    print(f"Checking {len(checks)} palette intervals by phase using {jobs} Python "
          f"{'process' if jobs == 1 else 'processes'}.", flush=True)

    def report(counts):
        for number, _ in enumerate(counts, 1):
            if number % 50 == 0 or number == len(checks):
                print(f"Verified {number}/{len(checks)} palette interval checks.", flush=True)

    if jobs == 1:
        report(map(check_phase, checks))
    else:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn")) as pool:
            report(pool.map(check_phase, checks))
    print(f"Palette cap {total} / {SCALE}, phases {phases}", flush=True)
