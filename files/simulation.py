"""DNA 복제 돌연변이 시뮬레이션

복제 오류가 3단계 검증(염기 선택 -> 즉시 교정 -> 불일치 복구)을 거쳐
돌연변이가 되는 과정을 확률로 실험한다.

실험 1. 교정/복구 시스템을 켜고 끄면 돌연변이가 얼마나 달라지는가
실험 2. 돌연변이원(외재적 요인)에 노출되면 돌연변이가 얼마나 늘어나는가
실험 3. 세포 수가 늘수록 평균 돌연변이 수가 기댓값에 가까워지는가
"""
import math
import random

import matplotlib.pyplot as plt

# ---- 모델 상수 (문헌의 대략적인 값, 생물과 조건마다 다름) ----
P_SELECT = 1e-5    # 1단계: 염기 선택 중 오류가 생길 확률
P_PROOF = 1e-2     # 2단계: 오류가 즉시 교정을 통과할 확률
P_REPAIR = 1e-3    # 3단계: 오류가 불일치 복구를 통과할 확률
GENOME = 4_600_000  # 유전체 크기(염기쌍), 대장균 수준
EXACT_LIMIT = 200_000  # 기댓값이 이보다 작으면 정확히 시뮬레이션


def binomial(n, p, rng):
    """n번 시행, 성공 확률 p일 때 성공 횟수를 하나 뽑는다.

    기댓값이 작으면 '다음 성공까지의 간격'을 건너뛰며 정확하게 세고,
    기댓값이 크면 정규분포 근사를 사용해 계산 시간을 줄인다.
    """
    if n <= 0 or p <= 0:
        return 0
    if p >= 1:
        return n
    mean = n * p
    if mean <= EXACT_LIMIT:
        log_q = math.log1p(-p)
        count = 0
        pos = int(math.log(1 - rng.random()) / log_q)
        while pos < n:
            count += 1
            pos += int(math.log(1 - rng.random()) / log_q) + 1
        return count
    sd = math.sqrt(mean * (1 - p))
    return min(n, max(0, round(rng.gauss(mean, sd))))


def mutation_probability(mult=1, proofread=True, repair=True):
    """염기 하나가 돌연변이로 확정될 이론적 확률 P (곱셈정리)."""
    p1 = min(0.5, P_SELECT * mult)
    p2 = P_PROOF if proofread else 1.0
    p3 = P_REPAIR if repair else 1.0
    return p1 * p2 * p3


def simulate(cells, genome=GENOME, mult=1, proofread=True, repair=True, seed=None):
    """세포 cells개가 DNA를 한 번씩 복제할 때의 결과를 돌려준다."""
    rng = random.Random(seed)
    p1 = min(0.5, P_SELECT * mult)
    p2 = P_PROOF if proofread else 1.0
    p3 = P_REPAIR if repair else 1.0

    total = cells * genome
    errors = binomial(total, p1, rng)             # 1단계: 복제 오류 발생
    passed_proof = binomial(errors, p2, rng)      # 2단계: 교정을 통과한 오류
    mutations = binomial(passed_proof, p3, rng)   # 3단계: 복구까지 통과 = 돌연변이

    # 돌연변이가 생긴 세포 수 (한 세포에 여러 개 생길 수도 있음)
    if mutations <= EXACT_LIMIT:
        mutated_cells = len({rng.randrange(cells) for _ in range(mutations)})
    else:
        mutated_cells = round(cells * (1 - math.exp(-mutations / cells)))

    return {
        "errors": errors,
        "passed_proof": passed_proof,
        "mutations": mutations,
        "mutated_cells": mutated_cells,
        "expected": total * mutation_probability(mult, proofread, repair),
    }


def setup_font():
    """한글이 깨지지 않도록 설치된 한글 글꼴을 찾아 적용한다."""
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in names:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def experiment_conditions(ax, cells=100_000):
    conditions = [
        ("교정 ON\n복구 ON", True, True),
        ("교정 OFF\n복구 ON", False, True),
        ("교정 ON\n복구 OFF", True, False),
        ("교정 OFF\n복구 OFF", False, False),
    ]
    labels, sim, exp = [], [], []
    print("\n[실험 1] 교정/복구 시스템 ON/OFF")
    for label, proof, repair in conditions:
        r = simulate(cells, proofread=proof, repair=repair, seed=1)
        P = mutation_probability(proofread=proof, repair=repair)
        print(f"  {label.replace(chr(10), ' ')}: P={P:.1e}, 기댓값={r['expected']:.1f}, 실제={r['mutations']}")
        labels.append(label)
        sim.append(r["mutations"])
        exp.append(r["expected"])
    ax.bar(labels, sim, color="#1f8a5b", label="시뮬레이션")
    ax.plot(labels, exp, "o", color="#c8432f", label="기댓값 n·P")
    ax.set_yscale("log")
    ax.set_ylabel("돌연변이 수 (로그 눈금)")
    ax.set_title(f"실험 1. 검증 시스템의 효과 (세포 {cells:,}개)")
    ax.legend()


def experiment_mutagen(ax, cells=100_000):
    mults = [1, 10, 100, 1000]
    sim, exp = [], []
    print("\n[실험 2] 돌연변이원 노출 (교정/복구 ON)")
    for m in mults:
        r = simulate(cells, mult=m, seed=2)
        print(f"  ×{m}: 기댓값={r['expected']:.1f}, 실제={r['mutations']}")
        sim.append(r["mutations"])
        exp.append(r["expected"])
    labels = [f"×{m}" for m in mults]
    ax.bar(labels, sim, color="#2f6fb5", label="시뮬레이션")
    ax.plot(labels, exp, "o", color="#c8432f", label="기댓값 n·P")
    ax.set_yscale("log")
    ax.set_xlabel("복제 오류 확률 배수")
    ax.set_ylabel("돌연변이 수 (로그 눈금)")
    ax.set_title("실험 2. 돌연변이원의 효과")
    ax.legend()


def experiment_convergence(ax):
    # 검증 시스템을 모두 끄고 유전체를 10만 염기쌍으로 두면 P=1e-5, 세포당 기댓값은 1개
    sizes = [10, 30, 100, 300, 1000, 3000, 10_000, 30_000, 100_000]
    means = []
    print("\n[실험 3] 세포 수와 세포당 평균 돌연변이 수 (기댓값 1.0)")
    for i, c in enumerate(sizes):
        r = simulate(c, genome=100_000, proofread=False, repair=False, seed=10 + i)
        mean = r["mutations"] / c
        print(f"  세포 {c:>7,}개: 평균 {mean:.3f}")
        means.append(mean)
    ax.semilogx(sizes, means, "o-", color="#1f8a5b", label="시뮬레이션 평균")
    ax.axhline(1.0, color="#c8432f", linestyle="--", label="기댓값 n·P = 1")
    ax.set_xlabel("세포 수")
    ax.set_ylabel("세포당 평균 돌연변이 수")
    ax.set_title("실험 3. 표본이 클수록 기댓값에 가까워진다")
    ax.legend()


def main():
    setup_font()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    experiment_conditions(axes[0])
    experiment_mutagen(axes[1])
    experiment_convergence(axes[2])
    fig.tight_layout()
    fig.savefig("results.png", dpi=150)
    print("\n그래프를 results.png로 저장했습니다.")
    plt.show()


if __name__ == "__main__":
    main()
