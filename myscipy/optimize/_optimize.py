import warnings
from numpy import (asarray, sqrt)
import numpy as np
from ._linesearch import (line_search_wolfe1, line_search_wolfe2,
                          LineSearchWarning)
from myscipy.optimize._differentiable_functions import (ScalarFunction,
                                                        FD_METHODS)
from scipy.optimize import OptimizeResult
from scipy.optimize._optimize import (_check_unknown_options,
                                      _check_positive_definite,
                                      _LineSearchError,
                                      _call_callback_maybe_halt,
                                      _print_success_message_or_warn)

# 옵티마이저의 표준 상태 메시지
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                             'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.',
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}

# np.finfo(float).eps = 2.22e-16
# 부동소수점 연산에서 발생할 수 있는 최소 오차
_epsilon = sqrt(np.finfo(float).eps)


# 벡터의 노름을 계산하는 vecnorm 함수
def vecnorm(x, ord=2):
    # ord가 무한대인 경우 x의 절대값 중 최대값을 반환
    if ord == np.inf:
        return np.amax(np.abs(x))
    # ord가 음의 무한대인 경우 x의 절대값 중 최소값을 반환
    elif ord == -np.inf:
        return np.amin(np.abs(x))
    # 그 외의 경우 ord 노름을 계산하여 반환
    else:
        return np.sum(np.abs(x)**ord, axis=0)**(1.0 / ord)


# 스칼라 함수의 최소화를 위한 ScalarFunction 객체를 생성하는 _prepare_scalar_function 함수
def _prepare_scalar_function(fun, x0, jac=None, args=(), bounds=None,
                             epsilon=None, finite_diff_rel_step=None,
                             hess=None):
    # jac이 호출 가능한 객체인 경우 grad에 할당
    if callable(jac):
        grad = jac
    # jac이 FD_METHODS 중 하나인 경우 grad에 할당하고 epsilon을 None으로 설정
    elif jac in FD_METHODS:
        epsilon = None
        grad = jac
    # 그 외의 경우 epsilon을 None으로 설정하고 grad를 '2-point'로 설정
    else:
        grad = '2-point'
        epsilon = epsilon

    # 헤세 행렬 hess가 None인 경우 None을 반환하는 함수로 설정
    if hess is None:
        def hess(x, *args):
            return None

    # 경계가 없는 경우 (-np.inf, np.inf)로 설정
    if bounds is None:
        bounds = (-np.inf, np.inf)

    # ScalarFunction 객체 생성
    sf = ScalarFunction(fun, x0, args, grad, hess,
                        finite_diff_rel_step, bounds, epsilon=epsilon)

    return sf


# Wolfe 조건을 이용한 선형 탐색을 수행하는 _line_search_wolfe12 함수
def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval,
                         **kwargs):
    # extra_condition을 kwargs에서 추출하거나 없으면 None으로 설정
    extra_condition = kwargs.pop('extra_condition', None)

    # line_search_wolfe1 함수로 선형 탐색을 수행하고 결과를 ret에 할당
    ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval,
                             **kwargs)

    # ret[0]이 None이 아니고 extra_condition이 None이 아닌 경우
    if ret[0] is not None and extra_condition is not None:
        xp1 = xk + ret[0] * pk  # xk에서 pk 방향으로 ret[0]만큼 이동한 위치를 xp1에 할당
        # extra_condition이 True가 아닌 경우 ret[0]를 None으로 설정
        if not extra_condition(ret[0], xp1, ret[3], ret[5]):
            ret = (None,)

    # ret[0]이 None인 경우 line_search_wolfe2 함수로 선형 탐색을 수행하고 결과를 ret에 할당
    if ret[0] is None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', LineSearchWarning)
            kwargs2 = {}
            for key in ('c1', 'c2', 'amax'):
                if key in kwargs:
                    kwargs2[key] = kwargs[key]
            ret = line_search_wolfe2(f, fprime, xk, pk, gfk,
                                     old_fval, old_old_fval,
                                     extra_condition=extra_condition,
                                     **kwargs2)

    # ret[0]이 None인 경우 _LineSearchError 발생
    if ret[0] is None:
        raise _LineSearchError()

    return ret


# BFGS 알고리즘을 이용한 하나 이상의 변수의 스칼라 함수의 최소화하는 _minimize_bfgs 함수
def _minimize_bfgs(fun, x0, args=(), jac=None, callback=None,
                   gtol=1e-5, norm=np.inf, eps=_epsilon, maxiter=None,
                   disp=False, return_all=False, finite_diff_rel_step=None,
                   xrtol=0, c1=1e-4, c2=0.9,
                   hess_inv0=None, **unknown_options):
    # 옵션을 확인하고 hess_inv0가 양의 정부호인지 확인
    _check_unknown_options(unknown_options)
    _check_positive_definite(hess_inv0)
    retall = return_all

    # x0를 1차원 배열로 변환 및 maxiter가 None인 경우 x0의 길이 * 200을 maxiter로 설정
    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    # 스칼라 최소화를 위한 스칼라 함수 객체 생성
    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    # 스칼라 함수 객체의 fun, grad 속성을 f, myfprime에 할당
    f = sf.fun
    myfprime = sf.grad

    # 초기값 x0에서의 함수값과 그래디언트를 old_fval, gfk에 할당
    old_fval = f(x0)
    gfk = myfprime(x0)

    k = 0  # 반복 횟수
    N = len(x0)  # 초기 추정값 x0의 길이
    I = np.eye(N, dtype=int)  # N x N 단위행렬
    # 초기 헤세 행렬의 역행렬 Hk를 hess_inv0로 설정하거나 None인 경우 단위행렬로 설정
    Hk = I if hess_inv0 is None else hess_inv0

    # 이전 함수값을 old_old_fval에 할당
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    # 초기 추정값 x0을 xk에 할당
    xk = x0
    # retall이 True인 경우 초기 추정값 x0을 allvecs에 추가
    if retall:
        allvecs = [x0]  # 최적화 과정의 모든 추정값을 저장할 리스트
    warnflag = 0  # 경고 플래그 초기화
    # 현재 그래디언트 gfk의 노름을 계산
    gnorm = vecnorm(gfk, ord=norm)
    # gnorm이 gtol보다 크고 k가 maxiter보다 작은 동안 반복
    while (gnorm > gtol) and (k < maxiter):
        # 현제 그래디언트 gfk에 헤세 행렬 Hk를 곱한 값에 -를 붙인 pk 계산
        pk = -np.dot(Hk, gfk)  # 검색 방향
        # _line_search_wolfe12 함수를 이용해 최적의 스텝 사이즈 alpha_k 계산
        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                _line_search_wolfe12(f, myfprime, xk, pk, gfk,
                                     old_fval, old_old_fval, amin=1e-100,
                                     amax=1e100, c1=c1, c2=c2)
        # _line_search_wolfe12 함수에서 _LineSearchError 발생 시
        # warnflag를 2로 설정하고 최적화 과정 종료
        except _LineSearchError:
            warnflag = 2
            break

        # 스텝 사이즈 alpha_k와 검색 방향 pk를 곱한 값을 sk에 할당
        sk = alpha_k * pk
        # 현재 추정값 xk에 sk를 더한 값을 xkp1에 할당
        xkp1 = xk + sk

        # retall이 True인 경우 xkp1을 allvecs에 추가
        if retall:
            allvecs.append(xkp1)
        # 현재 추정값 xk를 xkp1로 업데이트
        xk = xkp1
        # 업데이트된 gfkp1이 None인 경우 xkp1에서의 그래디언트 계산
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)

        # 업데이트된 그래디언트 gfkp1과 이전 그래디언트 gfk의 차이를 yk에 할당
        yk = gfkp1 - gfk
        # 현재 그래디언트 gfk를 gfkp1로 업데이트
        gfk = gfkp1
        k += 1  # 반복 횟수 1 증가
        # 현재 추정값 xk와 함수값 old_fval을 OptimizeResult 객체로 생성하여
        # 중간 결과를 intermediate_result에 할당
        intermediate_result = OptimizeResult(x=xk, fun=old_fval)
        # callback 함수를 호출하고 반환값이 True인 경우 최적화 과정 종료
        if _call_callback_maybe_halt(callback, intermediate_result):
            break
        # 현재 그래디언트 gfk의 노름을 계산
        gnorm = vecnorm(gfk, ord=norm)
        # gnorm이 gtol보다 작거나 같은 경우 최적화 과정 종료
        if (gnorm <= gtol):
            break

        # 스텝 사이즈 alpha_k와 검색 방향 pk를 곱한 값이
        # xrtol*(xrtol + vecnorm(xk))보다 작거나 같은 경우 최적화 과정 종료
        # 이는 xk의 변화량이 너무 작아서 최적화 과정이 더 이상 진행되지 않는 경우
        if (alpha_k*vecnorm(pk) <= xrtol*(xrtol + vecnorm(xk))):
            break

        # 현재 함수값 old_fval이 무한대인 경우 warnflag를 2로 설정하고 최적화 과정 종료
        if not np.isfinite(old_fval):
            warnflag = 2
            break

        # 그래디언트의 변화량 yk와 검색 방향의 변화량 sk의 내적을 계산
        rhok_inv = np.dot(yk, sk)
        # rhok_inv가 0인 경우 rhok를 1000으로 설정
        # 이는 yk와 sk가 수직인 경우로 헤세 행렬 업데이트를 수행하지 않음
        if rhok_inv == 0.:
            rhok = 1000.0
            # disp가 True인 경우 Divide-by-zero 경고 메시지 출력
            if disp:
                msg = 'Divide-by-zero encountered: rhok assumed large'
                _print_success_message_or_warn(True, msg)
        # rhok_inv가 0이 아닌 경우 rhok를 rhok_inv의 역수로 설정
        else:
            rhok = 1. / rhok_inv

        # A1은 단위행렬 I에서 sk의 외적과 yk의 외적을 곱한 값에 rhok를 곱한 값을 뺀 값
        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        # A2는 단위행렬 I에서 yk의 외적과 sk의 외적을 곱한 값에 rhok를 곱한 값을 뺀 값
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        # 헤세 행렬의 역행렬 Hk를 업데이트
        # Hk와 A2를 곱한 값에 A1을 곱한 후 sk와 sk의 외적을 곱한 값에 rhok를 곱한 값을 더함
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                           sk[np.newaxis, :])

    # 최적화 과정이 종료된 후 최종 함수값을 fval에 할당
    fval = old_fval

    # 최적화 과정의 상태 메시지 설정
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    # disp가 True일 경우 최적화 과정의 결과 출력
    if disp:
        _print_success_message_or_warn(warnflag, msg)
        print('         Current function value: %f' % fval)
        print('         Iterations: %d' % k)
        print('         Function evaluations: %d' % sf.nfev)
        print('         Gradient evaluations: %d' % sf.ngev)

    # 최적화 결과를 OptimizeResult 객체로 반환
    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)

    # retall이 True인 경우 result 딕셔너리에 allvecs 추가
    if retall:
        result['allvecs'] = allvecs
    return result
