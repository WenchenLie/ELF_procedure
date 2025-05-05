"""
Equivalent lateral force (ELF) method for the design of a frame structure in accordance with ASCE 7-22
"""
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def get_Cu(SD1: float) -> float:
    """Calculates Cu according to Table 12.8-1"""
    f = interp1d([0.1, 0.15, 0.2, 0.3, 0.4], [1.7, 1.6, 1.5, 1.4, 1.4], kind='linear', fill_value=(1.7, 1.4), bounds_error=False)
    return f(SD1)

def get_k(T: float) -> float:
    """Calculates k according to Section 12.8.3"""
    f = interp1d([0.5, 2.5], [1, 2], kind='linear', fill_value=(1, 2), bounds_error=False)
    return f(T)

def calc_ELF(
    Ct: float,
    x: float,
    wx: list | np.ndarray,
    hx: list | np.ndarray,
    S1: float,
    SDS: float,
    SD1: float,
    TL: float,
    R: float,
    Ie: float,
    output_file_name: str,
    load_pattern: Literal['as_calculated', 'inverted_triangle']='as_calculated'
) -> tuple[float, pd.DataFrame]:
    """Equivalent lateral force (ELF) method for the design of a frame structure in accordance with ASCE 7-22

    Args:
        Ct (float): Values of Approximate Period Parameters (Table 12.8-2)
        x (float): Values of Approximate Period Parameters (Table 12.8-2)
        wx (list | np.ndarray): Seismic weight of each story (from bottom to top)
        hx (list | np.ndarray): Height from base to level x (from bottom to top)
        S1 (float): Mapped maximum considered earthquake spectral response acceleration parameter (Section 11.4.2 or 11.4.4)
        SDS (float): Design spectral response acceleration parameter in the short period range (Section 11.4.5 or 11.4.8)
        SD1 (float): Design spectral response acceleration parameter at a period of 1.0 s (Section 11.4.5 or 11.4.6)
        TL (float): Long-period transition period (Section 11.4.6)
        R (float): Response modification factor (Table 12.2-1)
        Ie (float): Importance Factor (Table 1.5-2)
        output_file_name (str): Output file name
    """
    # Calculate fundamental period
    H = hx[-1]  # total height
    Ta = Ct * (H) ** x  # approximate period (12.8-8)
    get_Cu = interp1d([0.1, 0.15, 0.2, 0.3, 0.4], [1.7, 1.6, 1.5, 1.4, 1.4], kind='linear', fill_value=(1.7, 1.4), bounds_error=False)
    Cu = get_Cu(SD1)
    T = Cu * Ta  # fundamental period, section 12.8.2
    print(f"T = {T:.3f}")
    Cs = SDS / (R / Ie)  # seismic response coefficient (12.8-3)
    if T <= TL:
        Cs1 = SD1 / (T * (R / Ie))  # (12.8-4)
    else:
        Cs1 = SD1 * TL / (T ** 2 * (R / Ie))  # (12.8-5)
    Cs = min(Cs, Cs1)
    Cs2 = 0.044 * SDS * Ie  # Lower bounds for Cs (12.8-6)
    Cs = max(Cs, Cs2, 0.01)
    if S1 >= 0.6:
        Cs = max(Cs, 0.5 * S1 / (R / Ie))  # (12.8-7)
    print(f"Cs = {Cs:.3f}")
    W = np.sum(wx)  # total weight
    Vbase = Cs * W  # Seismic base shear (12.8-1)
    k = get_k(T)
    print(f"k = {k:.3f}")
    wx, hx = np.array(wx), np.array(hx)
    sum_ = np.sum(wx * hx ** k)
    if load_pattern == 'as_calculated':
        Cvx = wx * hx ** k / sum_  # (12.8-13)
    elif load_pattern == 'inverted_triangle':
        Cvx = hx / np.sum(hx)
    Fx = Cvx * Vbase  # (12.8-12)
    Vx = [np.sum(Fx[i:]) for i in range(len(hx))]
    Mbase = np.sum(Fx * hx)  # Base overturning moment
    df = pd.DataFrame({'Story': range(1, len(hx) + 1), 'wx': wx, 'hx': hx, 'wx*hx^k': wx * hx ** k, 'Cvx': Cvx, 'Fx': Fx, 'Vx': Vx})
    print(df)
    df.to_csv(f'output/{output_file_name}.csv', index=False)
    return Vbase, Mbase, df


if __name__ == "__main__":
    # 4-story
    # wx = np.array([281.596, 280.476, 280.474, 232.731]) * 10  # seismci weight in kN
    # hx = [4.3, 8.3, 12.3, 16.3]
    # 12-story
    wx = np.array([281.223, 280.476, 280.476, 280.476, 280.476, 280.476, 280.476, 280.476, 280.476, 280.476, 280.476, 232.731]) * 10
    hx = [4.2, 8.2, 12.2, 16.2, 20.2, 24.2, 28.2, 32.2, 36.2, 40.2, 44.2, 48.2]  # height from base to level x in m
    # 8-story
    # wx = np.array([281.223, 280.476, 280.476, 280.476, 280.476, 280.476, 280.476, 232.731]) * 10
    # hx = [4.2, 8.2, 12.2, 16.2, 20.2, 24.2, 28.2, 32.2]

    S1 = 0.84
    SDS = 1.44
    SD1 = 1.09
    TL = 8
    R = 8
    Ie = 1
    Ct, x = 0.0731, 0.75
    output_file_name = 'SCB-RC-12S'
    Vbase, Mbase, df = calc_ELF(Ct, x, wx, hx, S1, SDS, SD1, TL, R, Ie, output_file_name, load_pattern='as_calculated')
    print('---- Force demand ----')
    print(f"Vbase = {Vbase:.2f} kN")
    print(f"Mbase = {Mbase:.2f} kN-m")
    

