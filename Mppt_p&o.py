import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# =============
# 1. 초기 조건
# =============
V_INIT = 15.0
DELTA_V = 0.5

IRR_STC = 1000.0
TEMP_C = 25.0

P_MAX = 250.0
V_MP = 30.0
V_OC = 37.0
I_SC = 8.5

T_END = 10.0   # 총 시뮬레이션 시간 [s]

# 샘플링 시간 선택:
# 1번 조건 -> 0.01
# 2번 조건 -> 0.05
DT = 0.01

# 컨버터/시스템이 실제로 전압을 따라가는 속도를 위한 1차 동특성
TAU = 0.03  # [s]

# 실시간 그래프 업데이트 간격
PLOT_UPDATE_EVERY = max(1, int(0.03 / DT))

# 추가 옵션
SHOW_FINAL_DETAIL_PLOT = True     # 시뮬레이션 종료 후 고해상도 상세 그래프 표시
SAVE_FINAL_FIGURE = False         # 상세 그래프 PNG 저장 여부
FINAL_FIGURE_NAME = "po_mppt_detailed.png"


# ========================
# 2. 일사량 프로파일
#    0~3s   : 1000 W/m^2
#    3~6s   : 500 W/m^2
#    6~10s  : 1000 W/m^2
# ========================
def irradiance(t: float) -> float:
    if 0.0 <= t < 3.0:
        return 1000.0
    elif 3.0 <= t < 6.0:
        return 500.0
    else:
        return 1000.0


# ============================================================
# 3. Vmp=30V에서 정확히 최대전력이 나오도록 맞춘 PV 모델
# ============================================================
def solve_shape_exponent(vmp: float, voc: float, tol: float = 1e-12) -> float:
    """
    최대전력점이 Vmp에서 생기도록 하는 지수 a를 이분법으로 구함.
    조건:
        (Vmp/Voc)^a = 1 / (1 + a)
    """
    r = vmp / voc

    def f(a):
        return (r ** a) - (1.0 / (1.0 + a))

    lo, hi = 1e-8, 100.0
    flo, fhi = f(lo), f(hi)

    if flo * fhi > 0:
        raise RuntimeError("Exponent solving failed: invalid bracket.")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)

        if abs(fmid) < tol:
            return mid

        if flo * fmid < 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid

    return 0.5 * (lo + hi)


A_EXP = solve_shape_exponent(V_MP, V_OC)

# STC에서 P(Vmp)=Pmax가 되도록 스케일 조정
K_SCALE = P_MAX / (V_MP * I_SC * (1.0 - (V_MP / V_OC) ** A_EXP))


def pv_current(voltage: float, irr: float) -> float:
    """
    주어진 전압과 일사량에서 PV 전류 계산
    """
    v = np.clip(voltage, 0.0, V_OC)
    g_ratio = irr / IRR_STC

    current = K_SCALE * I_SC * g_ratio * (1.0 - (v / V_OC) ** A_EXP)

    if current < 0.0:
        current = 0.0

    return current


def pv_power(voltage: float, irr: float) -> float:
    """
    주어진 전압과 일사량에서 PV 전력 계산
    """
    return voltage * pv_current(voltage, irr)


# ============================================================
# 4. P&O MPPT 제어기
# ============================================================
class PerturbAndObserveMPPT:
    def __init__(self, v_init: float, delta_v: float, v_min: float = 0.0, v_max: float = V_OC):
        self.v_ref = float(v_init)
        self.delta_v = float(delta_v)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        self.prev_v = None
        self.prev_p = None
        self.direction = +1.0  # 시작 perturb 방향 (+)

    def update(self, measured_v: float, measured_i: float) -> float:
        """
        P&O 알고리즘 업데이트
        입력:
            measured_v : 현재 측정 전압
            measured_i : 현재 측정 전류
        출력:
            다음 전압 기준값 v_ref
        """
        p = measured_v * measured_i

        # 첫 샘플
        if self.prev_v is None or self.prev_p is None:
            self.prev_v = measured_v
            self.prev_p = p
            self.v_ref = np.clip(self.v_ref + self.direction * self.delta_v, self.v_min, self.v_max)
            return self.v_ref

        dV = measured_v - self.prev_v
        dP = p - self.prev_p

        # 고전적 P&O 로직
        if dP > 0:
            if dV > 0:
                self.direction = +1.0
            elif dV < 0:
                self.direction = -1.0
        elif dP < 0:
            if dV > 0:
                self.direction = -1.0
            elif dV < 0:
                self.direction = +1.0

        self.v_ref = np.clip(self.v_ref + self.direction * self.delta_v, self.v_min, self.v_max)

        self.prev_v = measured_v
        self.prev_p = p

        return self.v_ref


# ============================================================
# 5. 그래프 스타일 함수
# ============================================================
def style_axis(ax, y_label=None, x_label=None, x_major=1.0, x_minor=0.2):
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=11)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=11)

    ax.grid(True, which="major", linewidth=0.8, alpha=0.9)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.5)
    ax.minorticks_on()
    ax.xaxis.set_major_locator(MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor))

    # 일사량 변화 시점 표시
    ax.axvline(3.0, linestyle="--", linewidth=1.0)
    ax.axvline(6.0, linestyle="--", linewidth=1.0)


# ============================================================
# 6. 상세 결과 그래프
# ============================================================
def plot_final_results(results):
    t = results["time"]
    irr = results["irradiance"]
    v_ref = results["v_ref"]
    v_pv = results["v_pv"]
    i_pv = results["i_pv"]
    i_mpp = results["i_mpp"]
    p_pv = results["p_pv"]
    p_mpp = results["p_mpp"]
    eff = results["efficiency"]

    fig, axes = plt.subplots(5, 1, figsize=(16, 15), sharex=True)

    # 1) Irradiance
    axes[0].plot(t, irr, linewidth=2.2, label="Irradiance")
    axes[0].set_ylim(0, 1100)
    axes[0].legend(loc="upper right")
    style_axis(axes[0], y_label="Irr [W/m²]")

    # 2) Voltage
    axes[1].plot(t, v_pv, linewidth=2.0, label="PV Voltage")
    axes[1].plot(t, v_ref, "--", linewidth=1.6, label="Vref")
    axes[1].axhline(V_MP, linestyle=":", linewidth=1.5, label="Vmp=30V")
    axes[1].set_ylim(0, V_OC + 2)
    axes[1].legend(loc="upper right")
    style_axis(axes[1], y_label="Voltage [V]")

    # 3) Current
    axes[2].plot(t, i_pv, linewidth=2.0, label="PV Current")
    axes[2].plot(t, i_mpp, "--", linewidth=1.6, label="Ideal MPP Current")
    axes[2].set_ylim(0, I_SC + 1.0)
    axes[2].legend(loc="upper right")
    style_axis(axes[2], y_label="Current [A]")

    # 4) Power
    axes[3].plot(t, p_pv, linewidth=2.0, label="PV Power")
    axes[3].plot(t, p_mpp, "--", linewidth=1.6, label="Ideal MPP Power")
    axes[3].set_ylim(0, P_MAX + 30)
    axes[3].legend(loc="upper right")
    style_axis(axes[3], y_label="Power [W]")

    # 5) Efficiency
    axes[4].plot(t, eff, linewidth=2.0, label="Tracking Efficiency")
    axes[4].set_ylim(0, 105)
    axes[4].legend(loc="upper right")
    style_axis(axes[4], y_label="Efficiency [%]", x_label="Time [s]")

    for ax in axes:
        ax.set_xlim(0, T_END)

    fig.suptitle(f"P&O MPPT Tracking - Detailed View  (dt={DT}s, step={DELTA_V}V)", fontsize=15)
    plt.tight_layout()

    if SAVE_FINAL_FIGURE:
        plt.savefig(FINAL_FIGURE_NAME, dpi=200, bbox_inches="tight")
        print(f"[Saved] {FINAL_FIGURE_NAME}")

    plt.show()


# ============================================================
# 7. 시뮬레이션 함수
# ============================================================
def run_simulation(dt: float = DT, real_time_plot: bool = True):
    n_steps = int(T_END / dt) + 1
    t_arr = np.linspace(0.0, T_END, n_steps)

    # 로그 저장 배열
    irr_arr = np.zeros(n_steps)
    v_ref_arr = np.zeros(n_steps)
    v_pv_arr = np.zeros(n_steps)
    i_pv_arr = np.zeros(n_steps)
    i_mpp_arr = np.zeros(n_steps)
    p_pv_arr = np.zeros(n_steps)
    p_mpp_arr = np.zeros(n_steps)
    eff_arr = np.zeros(n_steps)

    # 제어기 초기화
    mppt = PerturbAndObserveMPPT(v_init=V_INIT, delta_v=DELTA_V, v_min=0.0, v_max=V_OC)

    # 초기 상태
    v_pv = V_INIT
    v_ref = V_INIT

    # 실시간 플롯 준비
    if real_time_plot:
        plt.ion()
        fig, axes = plt.subplots(5, 1, figsize=(15, 14), sharex=True)

        line_irr, = axes[0].plot([], [], linewidth=2, label="Irradiance")
        axes[0].set_ylim(0, 1100)
        axes[0].legend(loc="upper right")
        style_axis(axes[0], y_label="Irr [W/m²]")

        line_vpv, = axes[1].plot([], [], linewidth=2, label="PV Voltage")
        line_vref, = axes[1].plot([], [], "--", linewidth=1.5, label="Vref")
        axes[1].axhline(V_MP, linestyle=":", linewidth=1.5, label="Vmp=30V")
        axes[1].set_ylim(0, V_OC + 2)
        axes[1].legend(loc="upper right")
        style_axis(axes[1], y_label="Voltage [V]")

        line_ipv, = axes[2].plot([], [], linewidth=2, label="PV Current")
        line_impp, = axes[2].plot([], [], "--", linewidth=1.5, label="Ideal MPP Current")
        axes[2].set_ylim(0, I_SC + 1.0)
        axes[2].legend(loc="upper right")
        style_axis(axes[2], y_label="Current [A]")

        line_ppv, = axes[3].plot([], [], linewidth=2, label="PV Power")
        line_pmpp, = axes[3].plot([], [], "--", linewidth=1.5, label="Ideal MPP Power")
        axes[3].set_ylim(0, P_MAX + 30)
        axes[3].legend(loc="upper right")
        style_axis(axes[3], y_label="Power [W]")

        line_eff, = axes[4].plot([], [], linewidth=2, label="Tracking Efficiency")
        axes[4].set_ylim(0, 105)
        axes[4].legend(loc="upper right")
        style_axis(axes[4], y_label="Efficiency [%]", x_label="Time [s]")

        fig.suptitle(f"P&O MPPT Real-Time Tracking  (dt={dt}s, step={DELTA_V}V)", fontsize=14)
        plt.tight_layout()

    # 메인 루프
    for k, t in enumerate(t_arr):
        irr = irradiance(t)

        # 현재 PV 측정값
        i_pv = pv_current(v_pv, irr)
        p_pv = v_pv * i_pv

        # 이상적 MPP 전력 / 전류
        p_mpp = P_MAX * (irr / IRR_STC)
        i_mpp = p_mpp / V_MP if V_MP > 1e-12 else 0.0

        # 추종 효율
        eff = 100.0 * p_pv / p_mpp if p_mpp > 1e-12 else 0.0

        # 로그 저장
        irr_arr[k] = irr
        v_ref_arr[k] = v_ref
        v_pv_arr[k] = v_pv
        i_pv_arr[k] = i_pv
        i_mpp_arr[k] = i_mpp
        p_pv_arr[k] = p_pv
        p_mpp_arr[k] = p_mpp
        eff_arr[k] = eff

        # MPPT 제어기 업데이트
        v_ref = mppt.update(measured_v=v_pv, measured_i=i_pv)

        # 시스템/컨버터 전압 동특성
        alpha = dt / TAU
        alpha = np.clip(alpha, 0.0, 1.0)
        v_pv = v_pv + alpha * (v_ref - v_pv)
        v_pv = np.clip(v_pv, 0.0, V_OC)

        # 실시간 플롯 갱신
        if real_time_plot and (k % PLOT_UPDATE_EVERY == 0 or k == n_steps - 1):
            line_irr.set_data(t_arr[:k+1], irr_arr[:k+1])

            line_vpv.set_data(t_arr[:k+1], v_pv_arr[:k+1])
            line_vref.set_data(t_arr[:k+1], v_ref_arr[:k+1])

            line_ipv.set_data(t_arr[:k+1], i_pv_arr[:k+1])
            line_impp.set_data(t_arr[:k+1], i_mpp_arr[:k+1])

            line_ppv.set_data(t_arr[:k+1], p_pv_arr[:k+1])
            line_pmpp.set_data(t_arr[:k+1], p_mpp_arr[:k+1])

            line_eff.set_data(t_arr[:k+1], eff_arr[:k+1])

            for ax in axes:
                ax.set_xlim(0, T_END)

            plt.pause(0.001)

    if real_time_plot:
        plt.ioff()
        plt.show()

    return {
        "time": t_arr,
        "irradiance": irr_arr,
        "v_ref": v_ref_arr,
        "v_pv": v_pv_arr,
        "i_pv": i_pv_arr,
        "i_mpp": i_mpp_arr,
        "p_pv": p_pv_arr,
        "p_mpp": p_mpp_arr,
        "efficiency": eff_arr,
    }


# ============================================================
# 8. 결과 요약
# ============================================================
def print_summary(results):
    t = results["time"]
    v = results["v_pv"]
    i = results["i_pv"]
    p = results["p_pv"]
    eff = results["efficiency"]

    idx1 = (t >= 0.0) & (t < 3.0)
    idx2 = (t >= 3.0) & (t < 6.0)
    idx3 = (t >= 6.0) & (t <= 10.0)

    print("\n==================== Simulation Summary ====================")
    print(f"Sampling time dt        : {DT:.3f} s")
    print(f"Initial voltage V_init  : {V_INIT:.2f} V")
    print(f"Step size delta_V       : {DELTA_V:.2f} V")
    print(f"Temperature             : {TEMP_C:.1f} °C")
    print(f"PV specs                : Pmax={P_MAX:.1f}W, Vmp={V_MP:.1f}V, Voc={V_OC:.1f}V, Isc={I_SC:.1f}A")

    print("\n[0s ~ 3s, 1000 W/m²]")
    print(f"  Avg Voltage           : {np.mean(v[idx1]):.3f} V")
    print(f"  Avg Current           : {np.mean(i[idx1]):.3f} A")
    print(f"  Avg Power             : {np.mean(p[idx1]):.3f} W")
    print(f"  Avg Efficiency        : {np.mean(eff[idx1]):.3f} %")

    print("\n[3s ~ 6s, 500 W/m²]")
    print(f"  Avg Voltage           : {np.mean(v[idx2]):.3f} V")
    print(f"  Avg Current           : {np.mean(i[idx2]):.3f} A")
    print(f"  Avg Power             : {np.mean(p[idx2]):.3f} W")
    print(f"  Avg Efficiency        : {np.mean(eff[idx2]):.3f} %")

    print("\n[6s ~ 10s, 1000 W/m²]")
    print(f"  Avg Voltage           : {np.mean(v[idx3]):.3f} V")
    print(f"  Avg Current           : {np.mean(i[idx3]):.3f} A")
    print(f"  Avg Power             : {np.mean(p[idx3]):.3f} W")
    print(f"  Avg Efficiency        : {np.mean(eff[idx3]):.3f} %")

    print("\nFinal state")
    print(f"  Final PV Voltage      : {v[-1]:.3f} V")
    print(f"  Final PV Current      : {i[-1]:.3f} A")
    print(f"  Final PV Power        : {p[-1]:.3f} W")
    print(f"  Final Efficiency      : {eff[-1]:.3f} %")
    print("============================================================\n")


# ============================================================
# 9. 실행
# ============================================================
if __name__ == "__main__":
    results = run_simulation(dt=DT, real_time_plot=True)
    print_summary(results)

    if SHOW_FINAL_DETAIL_PLOT:
        plot_final_results(results)
