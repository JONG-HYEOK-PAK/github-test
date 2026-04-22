# INC알고리즘을 활용한 MPPT제어 시뮬레이션
# 2025/04/12/토
# 배재대학교 전기전자공학과 2146031 정승준

import numpy as np
import matplotlib.pyplot as plt

# ======================
# ===== 초기 조건 ======
# ======================

# 1. 알고리즘 제어 변수
V_init = 15             # 초기 전압(V)
Step_size = 0.5         # 스텝 크기/delta_V(V)

# 2. 환경 조건
irradiance = 1000       # 일사량(W/m^2)
temperature = 25        # 온도(섭씨(C))

# 3. 대상 패널 스펙
P_max = 250             # 최대 전력(W)
V_mp = 30               # 최대 전력점 전압(V)
V_oc = 37               # 개방 전압(V)
I_sc = 8.5              # 단락 전류(A)

# 일사량 변동
# 00:00.000 ~ 00:02:999 (안정기): 일사량 1000 W/m² 유지 (초기 목표점 도달 및 진동 확인)
# 00:03.000 ~ 00:05.999 (구름 통과): 일사량 500 W/m² 로 급락 (갑자기 줄어든 일사량에 대한 추종 속도 확인)
# 00:06.000 ~ 00:10.000 (다시 맑아짐): 일사량 1000 W/m² 로 급등 (P&O가 방향을 잃고 헤매는지, INC가 안정적으로 쫓아가는지 확인)


# ========================================================================
# irradiance_arr = np.array({0.0: 1000, 3.0: 500, 6.0: 1000})     # 일사량[초(sec), 일사량(W/m^2)]

V_pv = V_init                             # 현재 전압
I_pv = 0                                  # 현재 전류     * AI 사용
P_pv = I_pv * V_pv                        # 현재 전력
V_ref = V_init                            # 현재 목표 전압

V_old = 0         # 이전 전압
I_old = 0         # 이전 전류
P_old = 0         # 이전 전력

time_start = 0      # 시작시간
time_run = 10       # 작동시간
# time_run = 4      # 작동시간
time_step = 0.01    # 시간간격
time = np.arange(time_start * 1000, (time_run + time_step) * 1000, time_step * 1000) / 1000           # 부동 소수점 오류 주의

# AI 사용
V_arr = np.zeros(len(time))       # 전압
I_arr = np.zeros(len(time))       # 전류
P_arr = np.zeros(len(time))       # 전력


# =====================
# ===== PV panal ======
# =====================
class PvPanal:
  def __init__(self, irradiance, temperature, V_pv, I_pv, P_pv):
    self.irradiance = irradiance         # 일사량(W/m^2)
    self.temperature = temperature       # 온도(섭씨)

    self.V_pv = V_pv
    self.I_pv = I_pv
    self.P_pv = P_pv

  # 현재 전압 계산
  def PvVoltage(self):
    if (self.V_pv > V_oc):
      self.V_pv = V_oc        # max 값
    return self.V_pv      

# 현재 전류 계산
# =======AI 사용=========
  def PvCurrent(self):    
    alpha = 0.0005   # Isc 온도 계수
    beta  = -0.12    # Voc 온도 계수
    
    # 온도 + 일사량 반영
    I_scAct = I_sc * (self.irradiance / 1000) * (1 + alpha * (self.temperature - 25))
    V_ocAct = V_oc + beta * (self.temperature - 25)
    
    # 전류 계산
    # I = I_scAct * (1 - self.V_pv / V_ocAct)
   
    k = 8   # MPP 위치 조정용 지수
    I = I_scAct * (1 - (self.V_pv / V_ocAct) ** k)
    self.I_pv = max(I, 0)   # 음수 방지
    return self.I_pv  
# =======================

#현재 전력 계산
  def PvPower(self):
    self.P_pv = self.I_pv * self.V_pv

    if (self.P_pv > P_max):
      self.P_pv = P_max      # max 값
    return self.P_pv



# ======================
# ==== INC 알고리즘 ====
# ======================
class INC:
  def __init__ (self, aV_pv, aI_pv, aP_pv, aV_old, aI_old, aP_old, aV_ref, aStep_size):
    self.V_pv = aV_pv
    self.I_pv = aI_pv
    self.P_pv = aP_pv

    self.V_old = aV_old
    self.I_old = aI_old
    self.P_old = aP_old

    self.V_ref = aV_ref
    self.Step_size = aStep_size

    self.V_delta = aV_pv - aV_old                            # 전압 변화량
    self.I_delta = aI_pv - aI_old                            # 전류 변화량

    if (self.V_delta != 0):
      self.dIdV = self.I_delta / self.V_delta                  # 기울기
    else :
      self.dIdV = 0
    self.code = 0  

  def Vref (self):
    # if (self.V_delta > 0):
    #   if (self.I_delta == 0):ㅌ``
    #     self.V_ref
    #     self.code = 10
    #   elif (self.I_delta > 0):
    #     self.V_ref += self.Step_size
    #     self.code = 11  
    #   else:
    #     self.V_ref -= self.Step_size
    #     self.code = 12  

    # else:
    #   if (self.dIdV == -I_pv / V_pv):
    #     self.V_ref
    #     self.code = 20
    #   elif (self.dIdV > -I_pv / V_pv):
    #     self.V_ref += self.Step_size
    #     self.code = 21
    #   else :
    #     self.V_ref -= self.Step_size
    #     self.code = 22

    # =======AI 사용=========
    EPS = 1e-6
    if (self.V_delta == 0):
        if self.I_delta > 0:
            self.V_ref += self.Step_size
            self.code = 11
        elif self.I_delta < 0:
            self.V_ref -= self.Step_size
            self.code = 12
        else:         # self.I_delta == 0
            self.code = 13
    else:
        cond = -self.I_pv / self.V_pv

        # MPP 도달
        if abs(self.dIdV - cond) < EPS:
            self.code = 21

        elif self.dIdV > cond:
            self.V_ref += self.Step_size
            self.code = 22
        else:
            self.V_ref -= self.Step_size
            self.code = 23
    # ===========================
      
      
    if (self.V_ref > V_oc) :
      self.V_ref = V_oc
      self.code = 30

    return self.V_ref
  


# =======================
# ==== 시뮬레이션 시작 ====
# =======================
# plt.ion()
# fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# line_irr, = axes[0].plot([], [], linewidth=2, label="Irradiance")
# axes[0].set_ylabel("PV Power")
# axes[0].set_xlabel("PV Voltage")
# axes[0].grid(True)
# axes[0].legend(loc="upper right")

# line_vref, = axes[1].plot([], [], '--', linewidth=1.5, label="Vref")
# axes[1].axhline(V_mp, linestyle=':', linewidth=1.5, label="Vmp=30V")
# axes[1].set_ylabel("Voltage [V]")
# axes[1].grid(True)
# axes[1].legend(loc="upper right")

# line_ppv, = axes[2].plot([], [], '--', linewidth=1.5, label="Ideal MPP Power")
# axes[2].set_ylabel("Power [W]")
# axes[2].grid(True)
# axes[2].legend(loc="upper right")

# fig.suptitle(f"INC MPPT Real-Time Tracking  (dt={time_step}s, step={Step_size}V)", fontsize=14)
# plt.tight_layout()


V_old = V_arr[0] 
I_old = I_arr[0] 
P_old = P_arr[0] 


for i in range(1, len(time)):    # 실제 시뮬레이션 time for문   *0초는 제외!
# for i in range(0, 10, 1):        # 테스트 용 for문

  # 시간 변화에 따른 일사량
  if (0 <= time[i] and time[i] < 3):
    irradiance = 1000
  elif (3 <= time[i] and time[i] < 6):
    irradiance = 500
  else : 
    irradiance = 1000

  V_arr[i] = V_pv
  I_arr[i] = I_pv
  P_arr[i] = P_pv

  sp = PvPanal(irradiance, temperature, V_pv, I_pv, P_pv)           # irradiance, temperature, V_pv, I_pv, P_pv
  V_pv = sp.PvVoltage()      # 현재 전압
  I_pv = sp.PvCurrent()      # 현재 전류
  P_pv = sp.PvPower()        # 현재 전력

  Mppt = INC(V_pv, I_pv, P_pv, V_old, I_old, P_old, V_ref, Step_size)        # V_pv, I_pv, P_pv, V_old, I_old, P_old, V_ref, Step_size
  V_ref = Mppt.Vref()
  # print('INC code:', Mppt.code)                             # INC알고리즘 동작 확인 코드 
  

  V_old = V_arr[i - 1]      # 현재 전압       * i = 0일 경우  배열의 마지막 값이 들어가므로 주의
  I_old = I_arr[i - 1]      # 현재 전류
  P_old = P_arr[i - 1]      # 현재 전력
  V_pv = V_ref

  # line_irr.set_data(V_arr[:i], P_arr[:i])
  # line_vref.set_data(time[:i], V_arr[:i])
  # line_ppv.set_data(time[:i], P_arr[:i])
  
  # for ax in axes:
  #   ax.set_xlim(0, time_run)
  # axes[0].set_ylim(V_oc + 2, P_max + 30)
  # axes[1].set_ylim(0, V_oc + 2)
  # axes[2].set_ylim(0, P_max + 30)

  # plt.pause(0.001)
  
  # 확인용 print() 구문
  print(f"Time={time[i]:3f} ,G={irradiance}, Vpv={V_pv:.2f},  Vref={V_ref:.2f}, Ipv={I_pv:.2f}, Ppv = {P_pv:.2f}")    

# 확인용 print() 구문
# print('V_arr', V_arr)
# print('I_arr', I_arr)
# print('P_arr', P_arr)

# ==================================


# plt 실행
# 전압-전력 그래프
plt.subplot(3,1,1)
plt.plot(V_arr, P_arr, 'k.-')
plt.xlabel('Voltage')
plt.ylabel('Power')
plt.grid()

# 전압-전류 그래프
# plt.plot(V_arr, I_arr, 'k.-')
# plt.xlabel('Voltage')
# plt.ylabel('Current')
# plt.grid()

# # 시간-전압 그래프
plt.subplot(3,1,2)
plt.plot(time, V_arr, 'k.-')
plt.xlabel('time')
plt.ylabel('Voltage')
plt.grid()

# 시간-전류 그래프
# plt.plot(time, I_arr, 'k.-')
# plt.xlabel('time')
# plt.ylabel('Current')
# plt.grid()

plt.subplot(3,1,3)
plt.plot(time, P_arr, 'k.-')
plt.xlabel('time')
plt.ylabel('Power')
plt.grid()

# plt.ioff()
plt.show()
print('end')
