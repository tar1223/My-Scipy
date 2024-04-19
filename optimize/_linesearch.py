from scipy.optimize._dcsrch import DCSRCH
import numpy as np


def _check_c1_c2(c1, c2):
    if not (0 < c1 < c2 < 1):
        raise ValueError("'c1' and 'c2' do not satisfy"
                         "'0 < c1 < c2 < 1'.")


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

    # s만큼 pk 방향으로 이동한 지점에서의 함수값을 계산
    def phi(s):
        fc[0] += 1
        return f(xk + s*pk, *args)

    # s만큼 pk 방향으로 이동한 지점에서의 그레디언트를 구한 후 pk와 내적을 계산
    def derphi(s):
        gval[0] = fprime(xk + s*pk, *args)
        gc[0] += 1
        return np.dot(gval[0], pk)

    # 검색 방향 pk와 그레디언트 gfk의 내적을 derphi0에 할당
    derphi0 = np.dot(gfk, pk)

    # scalar_search_wolfe1 함수를 이용해 최적의 스텝 사이즈, 함수값, 이전 함수값을 계산
    stp, fval, old_fval = scalar_search_wolfe1(
        phi, derphi, old_fval, old_old_fval, derphi0,
        c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)

    return stp, fc[0], gc[0], fval, old_fval, gval[0]


# Wolfe 조건을 이용해 최적의 스텝 사이즈를 계산하는 scalar_search_wolfe1 함수
def scalar_search_wolfe1(phi, derphi, phi0=None, old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9,
                         amax=50, amin=1e-8, xtol=1e-14):
    # 0 < c1 < c2 < 1을 만족하는지 확인
    _check_c1_c2(c1, c2)

    # phi0, derphi0가 None인 경우 0에서의 함수값, 도함수값을 계산
    if phi0 is None:
        phi0 = phi(0.)
    if derphi0 is None:
        derphi0 = derphi(0.)

    # 초기 스텝 사이즈를 결정하는데 사용할 alpha1 계산
    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0

    # DCSRCH 객체를 이용해 최적의 스텝 사이즈, 함수값, 이전 함수값을 계산
    maxiter = 100

    dcsrch = DCSRCH(phi, derphi, c1, c2, xtol, amin, amax)
    stp, phi1, phi0, task = dcsrch(
        alpha1, phi0=phi0, derphi0=derphi0, maxiter=maxiter
    )

    return stp, phi1, phi0