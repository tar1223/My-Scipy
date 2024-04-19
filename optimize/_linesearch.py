from warnings import warn

from scipy.optimize._linesearch import scalar_search_wolfe2
from scipy.optimize._dcsrch import DCSRCH
import numpy as np


class LineSearchWarning(RuntimeWarning):
    pass


def _check_c1_c2(c1, c2):
    if not (0 < c1 < c2 < 1):
        raise ValueError("'c1' and 'c2' do not satisfy"
                         "'0 < c1 < c2 < 1'.")


# Wolfe 조건을 이용한 직선 탐색을 수행하는 line_search_wolfe1 함수
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


# Wolfe 조건을 이용한 직선 탐색을 수행하는 line_search_wolfe2 함수
def line_search_wolfe2(f, myfprime, xk, pk, gfk=None, old_fval=None,
                       old_old_fval=None, args=(), c1=1e-4, c2=0.9, amax=None,
                       extra_condition=None, maxiter=10):
    # 함수 계산 횟수, 그레디언트 계산 횟수, 그레디언트, 스텝 사이즈 초기화
    fc = [0]
    gc = [0]
    gval = [None]
    gval_alpha = [None]

    # alpha만큼 pk 방향으로 이동한 지점에서의 함수값을 계산
    def phi(alpha):
        gc[0] += 1
        return f(xk + alpha*pk, *args)

    fprime = myfprime

    # alpha만큼 pk 방향으로 이동한 지점에서의 그레디언트를 구한 후 pk와 내적을 계산
    def derphi(alpha):
        gc[0] += 1
        gval[0] = fprime(xk + alpha*pk, *args)
        gval_alpha[0] = alpha
        return np.dot(gval[0], pk)

    # gfk가 None인 경우 xk에서의 그래디언트 계산
    if gfk is None:
        gfk = fprime(xk, *args)
    # 검색 방향 pk와 그래디언트 gfk의 내적을 derphi0에 할당
    derphi0 = np.dot(gfk, pk)

    # extra_condition이 None이 아닌 경우 extra_condition2 함수 정의
    if extra_condition is not None:
        def extra_condition2(alpha, phi):
            # gval_alpha[0]이 alpha와 다른 경우 derphi 함수 실행
            if gval_alpha[0] != alpha:
                derphi(alpha)
            x = xk + alpha*pk  # xk에서 alpha만큼 pk 방향으로 이동한 지점을 x에 할당
            # alpha, x, phi, gval[0]으로 extra_condition 함수 실행
            return extra_condition(alpha, x, phi, gval[0])
    else:
        extra_condition2 = None

    # scalar_search_wolfe2 함수를 이용해 최적의 스텝 사이즈, 함수값, 이전 함수값, 그래디언트를 계산
    alpha_star, phi_star, old_fval, derphi_star = scalar_search_wolfe2(
        phi, derphi, old_fval, old_old_fval, derphi0, c1, c2, amax,
        extra_condition2, maxiter=maxiter
    )

    # derphi_star가 None인 경우 선형 탐색 알고리즘이 수렴하지 않았다는 경고 출력
    if derphi_star is None:
        warn('The line search algorithm did not converge',
             LineSearchWarning, stacklevel=2)
    # 그렇지 않은 경우 derphi_star에 gval[0]을 할당
    else:
        derphi_star = gval[0]

    return alpha_star, fc[0], gc[0], phi_star, old_fval, derphi_star
