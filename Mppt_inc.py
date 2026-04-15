# INC알고리즘을 활용한 MPPT제어 시뮬레이션
# 2025/04/12/토
# 배재대학교 전기전자공학과 2146031 정승준

import numpy as np
import math
import matplotlib.pyplot as plt

# ======================
# ===== 초기 조건 ======
# ======================

# 1. 알고리즘 제어 변수
V_init = 15             # 초기 전압(V)
Step_size = 0.5         # 스텝 크기/delta_V(V)
#

# 2. 환경 조건
Irradiance = 1000       # 일사량(W/m^2)
Temperature = 25        # 온도(섭씨(C))

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
# Irradiance_arr = np.array({0.0: 1000, 3.0: 500, 6.0: 1000})     # 일사량[초(sec), 일사량(W/m^2)]

V_pv = V_init                             # 현재 전압
I_pv = 0                                  # 현재 전류     * AI 사용
P_pv = I_pv * V_pv                        # 현재 전력

V_old = 0         # 이전 전압
I_old = 0         # 이전 전류
P_old = 0         # 이전 전력

V_ref = V_init

time_start = 0    # 시작시간
time_run = 10     # 작동시간
time_step = 0.05  # 시간간격
time = np.arange(time_start * 1000, time_run * 1000, time_step * 1000) / 1000           # 부동 소수점 오류 주의

# AI 사용
V_ref_arr = np.zeros(len(time))   # 목표 전압
V_arr = np.zeros(len(time))       # 전압
I_arr = np.zeros(len(time))       # 전류
P_arr = np.zeros(len(time))       # 전력


# =====================
# ===== PV panal ======
# =====================
class PvPanal:
  def __init__(self, Irradiance, Temperature, V_pv, I_pv, P_pv):
    self.Irradiance = Irradiance         # 일사량(W/m^2)
    self.Temperature = Temperature       # 온도(섭씨)

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
  # def pv_current(self, V, G = 1000):
  #   Isc = 8.5 * (G / 1000)
  #   Voc = 37.0
  #   self.I_pv = Isc * (1 - V / Voc)
  #   return self.I_pv

  def PvCurrent(self):    
    alpha = 0.0005   # Isc 온도 계수
    beta  = -0.12    # Voc 온도 계수
    
    # 온도 + 일사량 반영
    I_scAct = I_sc * (self.Irradiance / 1000) * (1 + alpha * (self.Temperature - 25))
    V_ocAct = V_oc + beta * (self.Temperature - 25)
    
    # 전류 계산
    I = I_scAct * (1 - self.V_pv / V_ocAct)
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
  def __init__ (self, I_pv, V_pv, P_pv, I_old, V_old, P_old, V_ref, Step_size):
    self.I_pv = I_pv
    self.V_pv = V_pv
    self.P_pv = P_pv

    self.I_old = I_old
    self.V_old = V_old
    self.P_old = P_old

    self.V_ref = V_ref
    self.Step_size = Step_size

    self.V_delta = V_pv - V_old                            # 전압 변화량
    if (self.V_delta == 0):
      self.V_delta = 0.01
    self.I_delta = I_pv - I_old                            # 전류 변화량
    
    # ----값 확인 print()----
    # print('Vpv', V_pv)
    # print('Vold', V_old)
    # print('Ipv', I_pv)
    # print('Iold', I_old)
    
    self.dIdV = self.I_delta / self.V_delta                # 기울기


  def main (self):
    if (self.V_delta > 0):
      if (self.I_delta == 0):
        self.V_ref, print(10)

      else:
        if (self.I_delta > 0):
          self.V_ref + self.Step_size, print(11)
        else:
          self.V_ref - self.Step_size, print(12)

    else:
      if (self.dIdV == -I_pv / -V_pv):
        self.V_ref, print(20)

      else:
        if (self.dIdV > -I_pv / -V_pv):
          self.V_ref + self.Step_size, print(21)
        else :
          self.V_ref - self.Step_size, print(22)
      
      
    if (self.V_ref > V_oc) :
      self.V_ref = V_oc, print(30)
    return self.V_ref


  

# =======================
# ==== 시뮬레이션 시작 ====
# =======================
# time = np.arange(time_start, time_run, time_step)

for i in range(len(time)):    # 실제 시뮬레이션 time for문
# for i in range(0, 3, 1):        # 테스트 용 for문

  # 시간 변화에 따른 일사량
  if (0 <= time[i] and time[i] < 3):
    irradiance = 1000
  elif (0 <= time[i] and time[i] < 3):
    irradiance = 500
  else : 
    irradiance = 1000


  Sp = PvPanal(Irradiance, Temperature, V_pv, I_pv, P_pv)           # Irradiance, Temperature, V_pv, I_pv, P_pv
  V_pv = Sp.PvVoltage()      # 현재 전압
  I_pv = Sp.PvCurrent()      # 현재 전류
  P_pv = Sp.PvPower()        # 현재 전력

  Mppt = INC(V_pv, I_pv, P_pv, V_old, I_old, P_old, V_ref, Step_size)        # I_pv, V_pv, P_pv, I_old, V_old, P_old, V_ref, Step_size
  V_ref = Mppt.main()
  
  V_old = V_pv      # 현재 전압
  I_old = I_pv      # 현재 전류
  P_old = P_pv      # 현재 전력

  V_arr[i] = V_pv
  V_pv = V_ref
  V_ref_arr[i] = V_ref

  I_arr[i] = I_pv
  P_arr[i] = P_pv

# 확인용 print() 구문
# print('V_arr', V_arr)
# print('I_arr', I_arr)
# print('P_arr', P_arr)

# ==================================


# 전압-전력 그래프
# plt 실행
plt.plot(V_arr, P_arr, 'k.-')
plt.xlabel('Voltage')
plt.ylabel('Power')
plt.grid()
plt.show()

print('end')
