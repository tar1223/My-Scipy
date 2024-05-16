from warnings import warn

from scipy.optimize._linesearch import _zoom
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


# Wolfe 조건을 이용해 최적의 스텝 사이즈를 계산하는 scalar_search_wolfe2 함수
def scalar_search_wolfe2(phi, derphi, phi0=None,
                         old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9, amax=None,
                         extra_condition=None, maxiter=10):
    # 0 < c1 < c2 < 1을 만족하는지 확인
    _check_c1_c2(c1, c2)

    # phi0가 None인 경우 xk에서의 함수값을 계산
    if phi0 is None:
        phi0 = phi(0.)

    # derphi0가 None인 경우 xk에서의 그레디언트를 계산
    if derphi0 is None:
        derphi0 = derphi(0.)

    # 초기 스텝 사이즈 alpha0을 0으로 초기화
    alpha0 = 0
    # old_phi0이 None이 아니고 derphi0이 0이 아닌 경우 alpha1을 계산
    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
    # 그렇지 않은 경우 alpha1을 1로 초기화
    else:
        alpha1 = 1.0

    # alpha1이 0보다 작은 경우 1로 설정
    if alpha1 < 0:
        alpha1 = 1.0

    # amax가 None이 아닌 경우 alpha1과 amax 중 작은 값을 alpha1로 설정
    if amax is not None:
        alpha1 = min(alpha1, amax)

    # alpha1만큼 pk 방향으로 이동한 지점에서의 함수값을 계산
    phi_a1 = phi(alpha1)

    # 초기 스텝 사이즈에 대한 함수값, 그레디언트 값을 phi_a0, derphi_a0에 할당
    phi_a0 = phi0
    derphi_a0 = derphi0

    # extra_condition이 None인 경우 True를 반환하는 함수 정의
    if extra_condition is None:
        def extra_condition(alpha, phi):
            return True

    # maxiter만큼 반복
    for i in range(maxiter):
        # alpha1이 0이거나 amax가 None이 아니고 alpha0이 amax와 같은 경우 에러 메시지 출력
        if alpha1 == 0 or (amax is not None and alpha0 == amax):
            alpha_star = None  # 최적의 스텝 사이즈를 None으로 설정
            phi_star = phi0  # 최적의 함수값을 phi0으로 설정
            phi0 = old_phi0  # 초기 함수값을 old_phi0으로 설정
            derphi_star = None  # 최적의 그레디언트를 None으로 설정

            if alpha1 == 0:
                msg = 'Rounding errors prevent the line search from converging'
            else:
                msg = 'The line search algorithm could not find ' + \
                      'a solution less than or equal to amax: %s' % amax

            warn(msg, LineSearchWarning, stacklevel=2)
            break

        # Armijo 조건을 만족하지 않거나
        # 첫 번째 반복이 아니면서 현재 스텝 사이즈에서의 함수값이 이전 스텝 사이즈에서의 함수값보다 큰 경우
        # 최적의 스텝 사이즈, 함수값, 그레디언트를 계산
        not_first_iteration = i > 0
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or \
           ((phi_a1 >= phi_a0) and not_first_iteration):
            alpha_star, phi_star, derphi_star = \
                _zoom(alpha0, alpha1, phi_a0,
                      phi_a1, derphi_a0, phi, derphi,
                      phi0, derphi0, c1, c2, extra_condition)
            break

        # 현재 스텝 사이즈에서의 그레디언트를 derphi_a1에 할당
        derphi_a1 = derphi(alpha1)
        # 현제 스텝 사이즈에서의 그레디언트가 초기 스텝 사이즈에서의 그레디언트의 -c2배보다 작은 경우
        if (abs(derphi_a1) <= -c2*derphi0):
            alpha_star = alpha1  # 최적의 스텝 사이즈를 alpha1로 설정
            phi_star = phi_a1  # 최적의 함수값을 phi_a1로 설정
            derphi_star = derphi_a1  # 최적의 그레디언트를 derphi_a1로 설정
            break

        # 현재 스텝 사이즈에서의 그레디언트가 0보다 큰 경우 최적의 스텝 사이즈, 함수값, 그레디언트를 계산
        if (derphi_a1 >= 0):
            alpha_star, phi_star, derphi_star = \
                _zoom(alpha1, alpha0, phi_a1,
                      phi_a0, derphi_a1, phi, derphi,
                      phi0, derphi0, c1, c2, extra_condition)
            break

        alpha2 = 2 * alpha1  # 현제 스텝 사이즈의 2배를 alpha2에 할당
        if amax is not None:  # amax가 None이 아닌 경우
            alpha2 = min(alpha2, amax)  # alpha2와 amax 중 작은 값을 alpha2로 설정
        alpha0 = alpha1  # 현제 스텝 사이즈를 이전 스텝 사이즈로 설정
        alpha1 = alpha2  # alpha2를 현제 스텝 사이즈로 설정
        phi_a0 = phi_a1  # 현제 스텝 사이즈에서의 함수값을 이전 스텝 사이즈에서의 함수값으로 설정
        phi_a1 = phi(alpha1)  # 현제 스텝 사이즈에서의 2배한 스텝 사이즈에서의 함수값을 계산
        derphi_a0 = derphi_a1  # 현제 스텝 사이즈에서의 그레디언트를 이전 스텝 사이즈에서의 그레디언트로 설정

    # maxiter만큼 반복 후 최적의 스텝 사이즈, 함수값, 이전 함수값, 그레디언트를 찾지 못한 경우 에러 메시지 출력
    else:
        alpha_star = alpha1  # 최적의 스텝 사이즈를 alpha1로 설정
        phi_star = phi_a1  # 최적의 함수값을 phi_a1로 설정
        derphi_star = None  # 최적의 그레디언트를 None으로 설정
        warn('The line search algorithm did not converge',
             LineSearchWarning, stacklevel=2)

    return alpha_star, phi_star, phi0, derphi_star
