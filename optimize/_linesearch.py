from scipy.optimize._linesearch import scalar_search_wolfe1
import numpy as np


# Wolfe 조건을 이용한 선형 탐색을 수행하는 line_search_wolfe1 함수
def line_search_wolfe1(f, fprime, xk, pk, gfk=None,
                       old_fval=None, old_old_fval=None,
                       args=(), c1=1e-4, c2=0.9, amax=50, amin=1e-8,
                       xtol=1e-14):
    # gfk가 None이면 fprime을 호출하여 xk에서의 그레디언트를 계산
    if gfk is None:
        gfk = fprime(xk, *args)

    # 그레디언트, 그레디언트 계산 횟수, 함수 계산 횟수 초기화
    gval = [gfk]
    gc = [0]
    fc = [0]

    # 스텝 사이즈 s에 대한 함수 값을 계산
    def phi(s):
        fc[0] += 1
        return f(xk + s*pk, *args)

    # 스텝 사이즈 s에 대한 함수의 도함수 값을 계산
    def derphi(s):
        gval[0] = fprime(xk + s*pk, *args)
        gc[0] += 1
        return np.dot(gval[0], pk)

    # 검색 방향 pk와 그레디언트 gfk의 내적을 derphi0에 할당
    derphi0 = np.dot(gfk, pk)

    # scalar_search_wolfe1 객체를 이용해 최적의 스텝 사이즈, 함수값, 이전 함수값을 계산
    stp, fval, old_fval = scalar_search_wolfe1(
        phi, derphi, old_fval, old_old_fval, derphi0,
        c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)

    return stp, fc[0], gc[0], fval, old_fval, gval[0]
