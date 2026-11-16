#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using I = std::int64_t;
using Wide = __int128_t;

constexpr I M = 100000000;
constexpr I SCALE = 1000000000000LL;
constexpr I BASE = 66521250;
constexpr I REFERENCE = 86485000;
constexpr I K_LO = 185480000;
constexpr I K_HI = 185481000;
constexpr I INTERCEPT = 1800000;
constexpr I SLOPE = 38500000;
constexpr I CLIP = 44600000;
constexpr I RAMP = 10000;
constexpr I TRIANGLE_INTERCEPT = 3775000;
constexpr I TRIANGLE_SLOPE = 35100000;
constexpr I TRIANGLE_MULTIPLIER = 10000000;
constexpr I TRIANGLE_MAX = 61700000;
constexpr I STAR_MAX = 61800000;

struct MomentBound {
    I star_weight, clique_weight;
    I intercept_low, slope_low, intercept_high, slope_high;
};

constexpr std::array<MomentBound,2> MOMENTS{{
    {18000000,8000000,6410000,28100000,6140000,29000000},
    {24000000,2500000,6865000,26800000,6321000,28500000}
}};

I gain_three(I s) {
    I root = static_cast<I>(std::sqrt(static_cast<long double>(M*s)));
    while (root*root > M*s)
        --root;
    while ((root+1)*(root+1) <= M*s)
        ++root;
    return Wide(s)*(3*M-root-1)/(6*M);
}

I tail_gain(I s) {
    if (s >= 77000000)
        return 30026000;
    if (s >= 70000000)
        return 8851000 + Wide(27500000)*s/M;
    if (s >= 62000000)
        return 6681000 + Wide(30600000)*s/M;
    if (s >= 53000000)
        return 5379000 + Wide(32700000)*s/M;
    if (s >= 43000000)
        return 2835000 + Wide(37500000)*s/M;
    return 0;
}

I sharper_tail_gain(I s) {
    if (s>=77000000) return 30151000;
    if (s>=70000000) return 8976000+Wide(27500000)*s/M;
    if (s>=66000000) return 7156000+Wide(30100000)*s/M;
    if (s>=60000000) return 6364000+Wide(31300000)*s/M;
    if (s>=54000000) return 5644000+Wide(32500000)*s/M;
    if (s>=50000000) return 3700000+Wide(36100000)*s/M;
    if (s>=43000000) return 2750000+Wide(38000000)*s/M;
    return 0;
}

I ceil_div(Wide numerator, Wide denominator) {
    assert(denominator > 0);
    if (numerator <= 0)
        return numerator / denominator;
    return (numerator + denominator - 1) / denominator;
}

I sqrt_floor(I x) {
    I root = static_cast<I>(std::sqrt(static_cast<long double>(x)));
    while (root * root > x)
        --root;
    while ((root + 1) * (root + 1) <= x)
        ++root;
    return root;
}

struct Line {
    I p, q;
    I ordinary, extra;
};

std::vector<Line> make_lines(const std::vector<I>& weights, const std::vector<I>& low) {
    std::vector<std::pair<I,I>> hull;
    for (std::size_t i = 0; i < low.size(); ++i) {
        const I w = weights[i], d = low[i];
        if (!hull.empty() && hull.back().first == w)
            continue;
        while (hull.size() > 1) {
            const auto [w0,d0] = hull[hull.size() - 2];
            const auto [w1,d1] = hull.back();
            if (Wide(d1 - d0) * (w - w1) < Wide(d - d1) * (w1 - w0))
                break;
            hull.pop_back();
        }
        hull.emplace_back(w, d);
    }
    std::vector<std::pair<I,I>> candidates{{0,1}};
    for (std::size_t i = 1; i < hull.size(); ++i)
        candidates.emplace_back(hull[i-1].second - hull[i].second,
                                hull[i].first - hull[i-1].first);
    I crossover_num = 0, crossover_den = 1;
    for (std::size_t i = 0; i < low.size(); ++i) {
        if (low[i] >= REFERENCE || weights[i] == M)
            continue;
        const I p = low[i] - REFERENCE, q = M - weights[i];
        if (Wide(p) * crossover_den < Wide(crossover_num) * q) {
            crossover_num = p;
            crossover_den = q;
        }
    }
    if (crossover_num)
        candidates.emplace_back(crossover_num, crossover_den);

    std::vector<Line> lines;
    for (auto [p,q] : candidates) {
        I a = q * (REFERENCE - low[0]) - p * weights[0];
        for (std::size_t i = 1; i < low.size(); ++i)
            a = std::max(a, q * (REFERENCE - low[i]) - p * weights[i]);
        lines.push_back({p, q, a, std::max<I>(0, -p * M - a)});
    }
    return lines;
}

struct Box {
    I degree_low, degree_high, fraction_low, fraction_high;
};

struct ProfileBound {
    I size, mass, density;
};

struct Phase {
    I smallest, width, smallest_squared, tolerance, threshold;
    int triangle_method;
    I triangle_bound = M, triangle_slope = M;
    std::vector<std::pair<I,I>> triangle_caps;
    const std::vector<std::pair<I,I>>& density_bounds;
    const std::vector<std::pair<I,I>>& bonus;
    std::vector<Line> lines;
    I boxes = 0;

    Phase(const std::vector<I>& weights, const std::vector<I>& low,
          const std::vector<std::pair<I,I>>& bonus, I tolerance,
          const std::vector<std::pair<I,I>>& density_bounds, int triangle_method)
        : smallest(weights.front()), width(M - smallest),
          smallest_squared(ceil_div(Wide(smallest) * smallest, M)),
          tolerance(tolerance), threshold(low.front()), triangle_method(triangle_method),
          density_bounds(density_bounds),
          bonus(bonus), lines(make_lines(weights, low)) {
        if (!triangle_method)
            return;
        const I base_slope=ceil_div(Wide(K_HI)*(REFERENCE-threshold),M);
        for (const auto& [intercept,slope] : density_bounds) {
            const I cap=intercept-slope+ceil_div(Wide(slope+base_slope)*threshold,M);
            if (cap<triangle_bound) {
                triangle_bound=cap;
                triangle_slope=slope+base_slope;
            }
        }
        triangle_caps.emplace_back(triangle_bound,triangle_slope);
        if (triangle_method<2)
            return;
        I beta=0;
        for (std::size_t i=1;i<weights.size();++i)
            beta=std::max(beta,ceil_div(Wide(M)*(weights[i]-smallest),low[i]-threshold));
        const I factor=ceil_div(Wide(M+smallest)*beta,M);
        if (Wide(factor)*M*M<=Wide(K_LO)*smallest*smallest) {
            triangle_bound=ceil_div(Wide(smallest)*smallest*triangle_bound,Wide(M)*M)
                +ceil_div(Wide(factor)*(M-REFERENCE)*threshold*(M-threshold),Wide(M)*M*M);
            triangle_slope=ceil_div(Wide(smallest)*smallest*triangle_slope,Wide(M)*M);
            triangle_caps.emplace_back(triangle_bound,triangle_slope);
        }
        if (triangle_method>=5) {
            const I slope=74000000;
            const I cap=scan_triangle(slope);
            triangle_caps.emplace_back(cap,slope);
            if (cap<triangle_bound) {
                triangle_bound=cap;
                triangle_slope=slope;
            }
        }
        if (triangle_method==7 || triangle_method>=9) {
            const I slope=98000000;
            const I cap=scan_triangle(slope);
            triangle_caps.emplace_back(cap,slope);
            if (cap+15000<triangle_bound) {
                triangle_bound=cap;
                triangle_slope=slope;
            }
        }
        if (triangle_method<9)
            triangle_caps={{triangle_bound,triangle_slope}};
    }

    I degree_sum_cap(I dl, I dh) const {
        const auto gain = [this](I d) {
            return ceil_div(Wide(K_HI)*((REFERENCE-threshold)*d
                                      + (M-REFERENCE)*(M-d)), Wide(M)*M);
        };
        const I gl=gain(dl), gh=gain(dh);
        I bound=BASE+std::max(gl,gh);
        for (const auto& [intercept,slope] : density_bounds) {
            bound=std::min(bound,intercept-slope
                +std::max(gl+ceil_div(Wide(slope)*dl,M),gh+ceil_div(Wide(slope)*dh,M)));
        }
        return bound;
    }

    ProfileBound profile(Box box) const {
        const auto [dl,dh,ul,uh] = box;
        const I mean_lo = smallest + Wide(width) * ul / M;
        const I mean_hi = smallest + ceil_div(Wide(width) * uh, M);
        const I t = ceil_div(Wide(dh) * mean_hi, M);
        I budget = ceil_div(Wide(dh) * dh, M);
        std::array<std::pair<I,I>,16> endpoints;
        for (std::size_t i=0;i<density_bounds.size();++i) {
            const auto [intercept,slope]=density_bounds[i];
            endpoints[i]={intercept-slope+ceil_div(Wide(slope)*dl,M),
                          intercept-slope+ceil_div(Wide(slope)*dh,M)};
        }
        for (const auto [p,q,a,b] : lines) {
            const Wide term = Wide(p) * mean_lo + a - b;
            const auto weighted_degree_term = [&](I d) {
                const Wide raw = Wide(M) * b + d * term;
                const I k = raw >= 0 ? K_HI : K_LO;
                return ceil_div(k * raw, Wide(q) * M * M);
            };
            const I low = weighted_degree_term(dl);
            const I high = weighted_degree_term(dh);
            budget = std::min(budget, BASE + std::max(low, high));
            for (std::size_t i=0;i<density_bounds.size();++i)
                budget=std::min(budget,std::max(low+endpoints[i].first,high+endpoints[i].second));
        }

        assert(budget >= 0);
        const I high_mass = ceil_div(Wide(dh) * uh, M);
        const I hh = std::min(budget, ceil_div(Wide(high_mass) * high_mass, M));
        const I um = std::clamp(M / 2, ul, uh);
        const I mixed = ceil_div(Wide(2) * dh * dh * um * (M - um), Wide(M) * M * M);
        const I cross = std::min(budget - hh, mixed);
        const I c_ref = ceil_div(Wide(hh) * M + Wide(cross) * smallest
                           + Wide(budget - hh - cross) * smallest_squared, M);
        return {t,budget,c_ref};
    }

    I scan_triangle(I slope) const {
        const I scan_tolerance=24000;
        const auto bound=[this,slope](Box box) -> I {
            const auto [t,budget,c]=profile(box);
            const I density=std::min(c,ceil_div(Wide(t)*t,M));
            return density-Wide(slope)*(box.degree_low-threshold)/M;
        };
        I estimate=0;
        for (I d:{threshold,M})
            for (I u:{I(0),M/4,M/2,3*M/4,M})
                estimate=std::max(estimate,bound({d,d,u,u}));
        std::vector<Box> stack{{threshold,M,0,width?M:0}};
        while (!stack.empty()) {
            const Box box=stack.back();
            stack.pop_back();
            const I cap=bound(box);
            if (cap<=estimate+scan_tolerance)
                continue;
            const auto [dl,dh,ul,uh]=box;
            const I d=(dl+dh)/2, u=(ul+uh)/2;
            estimate=std::max(estimate,bound({d,d,u,u}));
            if (cap<=estimate+scan_tolerance)
                continue;
            if (std::max(dh-dl,uh-ul)<=1) {
                estimate=std::max(estimate,cap);
            } else if (dh-dl>1 && (uh-ul<=1 || dh-dl>=Wide(uh-ul)*width/M)) {
                stack.push_back({dl,d,ul,uh});
                stack.push_back({d,dh,ul,uh});
            } else {
                assert(ul<u && u<uh);
                stack.push_back({dl,dh,ul,u});
                stack.push_back({dl,dh,u,uh});
            }
        }
        return estimate+scan_tolerance;
    }

    I hit_upper(Box box) {
        ++boxes;
        const auto [t,budget,c_ref]=profile(box);
        const auto [dl,dh,ul,uh]=box;
        const I c = std::min(c_ref, ceil_div(Wide(t) * t, M));

        const I s = M - c;
        assert(s >= bonus.front().first);
        I gain = INTERCEPT + Wide(SLOPE) * std::min(s, CLIP) / M;
        std::size_t i = bonus.size() - 1;
        while (s < bonus[i].first)
            --i;
        const auto [start,value] = bonus[i];
        if (i > 0 && s - start < RAMP) {
            const I previous = bonus[i-1].second;
            gain += previous + Wide(value - previous) * (s - start) / RAMP;
        } else {
            gain += value;
        }
        gain=std::max({gain,tail_gain(s),gain_three(s)});
        if (triangle_method>=8)
            gain=std::max(gain,sharper_tail_gain(s));
        const I s_ref=M-c_ref;
        const I difference=triangle_method && 25000000<=s_ref && s_ref<=48000000
            ? std::max<I>(0,degree_sum_cap(dl,dh)-budget) : 0;
        for (const auto& [cap_value,slope_value]:triangle_caps) {
            const I triangle_bound=cap_value,triangle_slope=slope_value;
            const I gap=std::max<I>(0,TRIANGLE_MAX-triangle_bound);
            const I increase=std::max<I>(0,triangle_bound-TRIANGLE_MAX);
            const I derivative=ceil_div(Wide(TRIANGLE_MULTIPLIER)*gap*std::max<I>(0,4*c_ref-M),Wide(M)*M);
            if (triangle_method && s_ref >= 25500000 && s_ref <= 48000000 &&
                Wide(TRIANGLE_SLOPE-derivative)*K_LO*smallest*smallest >=
                Wide(TRIANGLE_MULTIPLIER)*triangle_slope*c_ref*M) {
                const I correction=ceil_div(Wide(TRIANGLE_MULTIPLIER)*triangle_slope*c_ref*difference,
                                            Wide(M)*M*K_LO)
                    +ceil_div(Wide(TRIANGLE_MULTIPLIER)*increase,M);
                const I bonus=Wide(TRIANGLE_MULTIPLIER)*gap*c_ref*(2*c_ref-M)/(Wide(M)*M*M);
                const I triangular_gain=TRIANGLE_INTERCEPT+Wide(TRIANGLE_SLOPE)*s_ref/M-correction+bonus;
                gain=std::max(gain,std::min<I>(triangular_gain,20835000));
            }
            if (triangle_method>=3 && s_ref>=25000000 && s_ref<=42000000) {
                const I dgap=std::max<I>(0,STAR_MAX-triangle_bound);
                const I increase=std::max<I>(0,triangle_bound-STAR_MAX);
                const I root=sqrt_floor(M*c_ref)+1;
                const I c_cube=Wide(c_ref)*c_ref*c_ref/(Wide(M)*M);
                for (int i=0;i<std::min<int>(2,triangle_method-2);++i) {
                    const auto moment=MOMENTS[i];
                    const I factor=ceil_div(Wide(moment.star_weight)*t*t*t,Wide(M)*M*M)
                        +ceil_div(Wide(moment.clique_weight)*c_ref*root,Wide(M)*M);
                    const I breakpoint=i==0?30000000:32000000;
                    I line=s_ref<=breakpoint
                        ? moment.intercept_low+Wide(moment.slope_low)*s_ref/M
                        : moment.intercept_high+Wide(moment.slope_high)*s_ref/M;
                    I slope_min=moment.slope_low;
                    I remaining_gap=dgap;
                    if (i==1 && triangle_method>=6 && dgap>0) {
                        const I low_line=(s_ref<32000000)
                            ? 7600000+Wide(26800000)*s_ref/M
                            : 7760000+Wide(26300000)*s_ref/M;
                        assert(low_line>=line);
                        const I portion=std::min<I>(6000000,dgap);
                        line+=Wide(low_line-line)*portion/6000000;
                        remaining_gap-=portion;
                        slope_min=26300000;
                    }
                    const I derivative=ceil_div(Wide(3)*moment.star_weight*remaining_gap*c_ref*c_ref,
                                                Wide(M)*M*M);
                    if (Wide(slope_min-derivative)*K_LO*smallest*smallest
                        < Wide(triangle_slope)*factor*M*M)
                        continue;
                    const I correction=ceil_div(Wide(triangle_slope)*factor*difference,Wide(M)*K_LO)
                        +ceil_div(Wide(moment.star_weight+moment.clique_weight)*increase,M);
                    const I addition=Wide(moment.star_weight)*remaining_gap*c_cube/(Wide(M)*M);
                    gain=std::max(gain,std::min<I>(18711000,line+addition-correction));
                }
            }
        }
        const I x = M - t;
        const I exponential = M + x + Wide(x) * x / (2 * M)
            + Wide(x) * x * x / (Wide(6) * M * M)
            + Wide(x) * x * x * x / (Wide(24) * M * M * M);
        const I padded = M - Wide(gain) * exponential / M;
        const I s0 = std::max<I>(0, Wide(t) * t / M - c);
        const I square_root = sqrt_floor(M * s0) + 1;
        const I direct = t - Wide(s0) * (3 * M - square_root) / (6 * M);
        return std::min(padded, direct);
    }

    I maximize(I lower, I upper) {
        I estimate = 0;
        for (I degree : {lower, upper})
            for (I fraction : std::array<I,6>{0,M/10,M/5,M/2,9*M/10,M})
                estimate = std::max(estimate, hit_upper({degree, degree, fraction, fraction}));
        std::vector<Box> stack{{lower, upper, 0, width ? M : 0}};
        while (!stack.empty()) {
            const auto box = stack.back();
            stack.pop_back();
            const I bound = hit_upper(box);
            if (bound <= estimate + tolerance)
                continue;
            const auto [dl,dh,ul,uh] = box;
            const I d = (dl + dh) / 2, u = (ul + uh) / 2;
            estimate = std::max(estimate, hit_upper({d,d,u,u}));
            if (bound <= estimate + tolerance)
                continue;
            if (std::max(dh - dl, uh - ul) <= 1) {
                estimate = std::max(estimate, bound);
            } else if (dh - dl > 1 && (uh - ul <= 1 || dh - dl >= Wide(uh-ul)*width/M)) {
                stack.push_back({dl,d,ul,uh});
                stack.push_back({d,dh,ul,uh});
            } else {
                assert(ul < u && u < uh);
                stack.push_back({dl,dh,ul,u});
                stack.push_back({dl,dh,u,uh});
            }
        }
        return estimate + tolerance;
    }
};

int main() {
    std::size_t classes, count;
    I q, cutoff, tolerance, claim;
    std::size_t density_count;
    int use_triangles;
    if (!(std::cin >> classes >> q >> cutoff >> tolerance >> claim >> count >> density_count >> use_triangles))
        return 1;
    assert(classes >= 2 && q >= 2 && cutoff > 0 && tolerance > 0);
    assert(0<density_count && density_count<=16);
    std::vector<std::pair<I,I>> density_bounds(density_count);
    for (auto& [intercept,slope] : density_bounds) {
        std::cin >> intercept >> slope;
        assert(0 < intercept && intercept <= M && slope >= 0);
    }
    std::vector<I> lower(classes);
    for (I& x : lower)
        std::cin >> x;
    assert(std::adjacent_find(lower.begin(),lower.end(),std::greater_equal<I>())==lower.end());
    assert(lower.back()<M);
    assert(lower.front() <= REFERENCE);
    assert(0<=use_triangles && use_triangles<=10);

    std::vector<std::pair<I,I>> bonus(count);
    for (auto& [s,b] : bonus)
        std::cin >> s >> b;
    assert(!bonus.empty());
    assert(BASE + ceil_div(Wide(K_HI) * (REFERENCE - lower.front()), M)
           < M - bonus.front().first);
    for (std::size_t i = 1; i < bonus.size(); ++i) {
        assert(bonus[i-1].first + RAMP < bonus[i].first);
        assert(bonus[i-1].second <= bonus[i].second);
    }

    std::vector<I> weights(classes, M);
    I scale = SCALE, palette = 0, phases = 0, boxes = 0;
    while (scale > cutoff) {
        assert(weights.back() == M && std::is_sorted(weights.begin(), weights.end()));
        I phase_q=q, phase_tolerance=tolerance;
        if (use_triangles>=10) {
            if (scale<SCALE/50) {
                phase_q=std::min<I>(q,45);
                phase_tolerance=4*tolerance;
            } else if (scale<SCALE/10) {
                phase_q=std::min<I>(q,85);
                phase_tolerance=2*tolerance;
            } else if (scale<2*SCALE/5) {
                phase_q=std::min<I>(q,185);
            }
        }
        Phase phase(weights, lower, bonus, phase_tolerance, density_bounds, use_triangles);
        std::vector<I> caps(classes);
        I phase_boxes=0;
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:phase_boxes) schedule(dynamic)
        #endif
        for (std::size_t i=0;i<classes;++i) {
            auto local=phase;
            caps[i]=local.maximize(lower[i],i+1<classes?lower[i+1]:M);
            phase_boxes+=local.boxes;
        }
        for (std::size_t i=1;i<classes;++i)
            caps[i]=std::max(caps[i],caps[i-1]);
        boxes+=phase_boxes;
        const I peak = caps.back();
        palette += ceil_div(Wide(scale) * peak, Wide(M) * phase_q);
        for (std::size_t i = 0; i < classes; ++i) {
            assert(caps[i] > 0 && caps[i] <= peak && peak < 2 * caps[i]);
            const I p=caps[i];
            if (use_triangles>=8) {
                const I r=peak-p;
                const Wide denominator=Wide(2)*phase_q*phase_q*p*p;
                const Wide numerator=denominator-Wide(2)*phase_q*p*r-Wide(r)*(p-r);
                weights[i]=ceil_div(Wide(weights[i])*numerator,denominator);
            } else {
                weights[i]=ceil_div(Wide(weights[i])*((phase_q+1)*p-peak),Wide(phase_q)*p);
            }
        }
        const I smaller = ceil_div(Wide(scale) * (phase_q - 1), phase_q);
        assert(smaller < scale);
        scale = smaller;
        ++phases;
        if (phases % 100 == 0)
            std::cerr << "Verified " << phases << " phases\n";
    }
    Phase finish(weights, lower, bonus, tolerance, density_bounds, use_triangles);
    I finish_cap = 0;
    for (std::size_t i = 0; i < classes; ++i)
        finish_cap = std::max(finish_cap,
            finish.maximize(lower[i], i + 1 < classes ? lower[i+1] : M));
    boxes += finish.boxes;
    const I total = palette + ceil_div(Wide(scale) * finish_cap, M);
    std::cout << "Palette cap " << total << " / " << SCALE
              << ", phases " << phases << ", boxes " << boxes << '\n';
    if (claim > 0)
        assert(total < claim && Wide(claim) * M <= Wide(lower.front()) * SCALE);
}
