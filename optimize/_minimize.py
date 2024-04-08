from warnings import warn

import numpy as np

from scipy.optimize._minimize import (standardize_constraints,
                                      standardize_bounds, _validate_bounds,
                                      _optimize_result_for_equal_bounds,
                                      _remove_from_bounds, _remove_from_func,
                                      _add_to_array)

# 제약 없는 최소화
from scipy.optimize._optimize import (_minimize_neldermead, _minimize_powell,
                                      _minimize_cg, _minimize_bfgs,
                                      _minimize_newtoncg, MemoizeJac,
                                      _wrap_callback)
from scipy.optimize._trustregion_dogleg import _minimize_dogleg
from scipy.optimize._trustregion_ncg import _minimize_trust_ncg
from scipy.optimize._trustregion_krylov import _minimize_trust_krylov
from scipy.optimize._trustregion_exact import _minimize_trustregion_exact
from scipy.optimize._trustregion_constr import _minimize_trustregion_constr

# 제약 있는 최소화
from scipy.optimize._lbfgsb_py import _minimize_lbfgsb
from scipy.optimize._tnc import _minimize_tnc
from scipy.optimize._cobyla_py import _minimize_cobyla
from scipy.optimize._slsqp_py import _minimize_slsqp
from scipy.optimize._differentiable_functions import FD_METHODS


# 하나 이상의 변수에 대한 스칼라 함수의 최소화 문제를 해결하는 minimize 함수
def minimize(fun, x0, args=(), method=None, jac=None, hess=None,
             hessp=None, bounds=None, constraints=(), tol=None,
             callback=None, options=None):
    '''
    fun: 최소화할 목적 함수
    x0: 최적화를 시작할 초기 추정치
    args: fun에 전달할 추가 인수
    method: 최적화 방법
    jac: 목적 함수의 자코비안(그레디언트) 벡터
    hess: 목적 함수의 헤시안 행렬
    hessp: 목적 함수의 헤시안-벡터 곱
    bounds: 변수의 경계 조건
    constraints: 제약 조건
    tol: 수치적 최적화 알고리즘의 수렴 기준
    callback: 최적화 알고리즘의 반복마다 호출되는 함수
    options: 최적화 방법에 대한 추가 옵션
    '''

    # x0가 1차원 배열이 아닌 경우 1차원 배열로 변환
    x0 = np.atleast_1d(np.asarray(x0))

    # x0가 1차원 배열이 아닌 경우 예외 처리
    if x0.ndim != 1:
        raise ValueError("'x0' must only have one dimension.")

    # x0의 dtype이 정수인 경우 float로 변환
    if x0.dtype.kind in np.typecodes['AllInteger']:
        x0 = np.asarray(x0, dtype=float)

    # args가 튜플이 아닌 경우 튜플로 변환
    if not isinstance(args, tuple):
        args = (args,)

    # method가 None인 경우 최적화 방법을 자동으로 선택
    if method is None:
        if constraints:
            method = 'SLSQP'
        elif bounds is not None:
            method = 'L-BFGS-B'
        else:
            method = 'BFGS'

    # method가 호출 가능한 함수인 경우 '_custom'으로 설정
    # 그렇지 않은 경우 method를 소문자로 변환
    if callable(method):
        meth = '_custom'
    else:
        meth = method.lower()

    # 선택 지정 가능한 매개변수가 method에 지원되지 않는 경우 경고 메시지 출력
    if options is None:
        options = {}
    # - jac
    if meth in ('nelder-mead', 'powell', 'cobyla') and bool(jac):
        warn('Method %s does not use gradient information (jac).' % method,
             RuntimeWarning, stacklevel=2)
    # - hess
    if meth not in ('newton-cg', 'dogleg', 'trust-ncg', 'trust-constr',
                    'trust-krylov', 'trust-exact', '_custom') \
            and hess is not None:
        warn('Method %s does not use Hessian information (hess).' % method,
             RuntimeWarning, stacklevel=2)
    # - hessp
    if meth not in ('newton-cg', 'trust-ncg', 'trust-constr',
                    'trust-krylov', '_custom') \
            and hessp is not None:
        warn('Method %s does not use Hessian-vector product '
             'information (hessp).' % method,
             RuntimeWarning, stacklevel=2)
    # - cnstraints 또는 bounds
    if (meth not in ('cobyla', 'slsqp', 'trust-constr', '_custom') and
            np.any(constraints)):
        warn('Method %s cannot handle constraints.' % method,
             RuntimeWarning, stacklevel=2)
    if meth not in ('nelder-mead', 'powell', 'l-bfgs-b', 'cobyla', 'slsqp',
                    'tnc', 'trust-constr', '_custom') and bounds is not None:
        warn('Method %s cannot handle bounds.' % method,
             RuntimeWarning, stacklevel=2)
    # - return_all
    if (meth in ('l-bfgs-b', 'tnc', 'cobyla', 'slsqp') and
            options.get('return_all', False)):
        warn('Method %s does not support the return_all option.' % method,
             RuntimeWarning, stacklevel=2)

    # 자코비안(그레디언트) 벡터 확인
    # jac이 호출 가능한 함수인 경우 그대로 사용
    if callable(jac):
        pass
    # jac이 True인 경우 fun을 MemoizeJac으로 래핑
    elif jac is True:
        fun = MemoizeJac(fun)
        jac = fun.derivative
    # jqc이 '2-point', '3-point', 'cs'이고
    # meth가 'trust-constr', 'bfgs', 'cg', 'l-bfgs-b', 'tnc', 'slsqp'인 경우 그대로 사용
    elif (jac in FD_METHODS and
          meth in ['trust-constr', 'bfgs', 'cg', 'l-bfgs-b', 'tnc', 'slsqp']):
        pass
    # meth가 'trust-constr'인 경우 jac을 '2-point'로 설정
    elif meth in ['trust-constr']:
        jac = '2-point'
    # jac이 False인 경우 None으로 설정
    elif jac is None or bool(jac) is False:
        jac = None
    # 그렇지 않은 경우 jac을 None으로 설정
    else:
        jac = None

    # 기본 허용 오차(tol) 설정
    if tol is not None:
        options = dict(options)
        if meth == 'nelder-mead':
            options.setdefault('xatol', tol)
            options.setdefault('fatol', tol)
        if meth in ('newton-cg', 'powell', 'tnc'):
            options.setdefault('xtol', tol)
        if meth in ('powell', 'l-bfgs-b', 'tnc', 'slsqp'):
            options.setdefault('ftol', tol)
        if meth in ('bfgs', 'cg', 'l-bfgs-b', 'tnc', 'dogleg',
                    'trust-ncg', 'trust-exact', 'trust-krylov'):
            options.setdefault('gtol', tol)
        if meth in ('cobyla', '_custom'):
            options.setdefault('tol', tol)
        if meth == 'trust-constr':
            options.setdefault('xtol', tol)
            options.setdefault('gtol', tol)
            options.setdefault('barrier_tol', tol)

    # meth가 '_custom'이라면 사용자 정의 객채를 사용하여 최적화
    if meth == '_custom':
        return method(fun, x0, args=args, jac=jac, hess=hess, hessp=hessp,
                      bounds=bounds, constraints=constraints,
                      callback=callback, **options)

    # constraints 표준화
    constraints = standardize_constraints(constraints, x0, meth)

    remove_vars = False
    if bounds is not None:
        # 경계를 표준화하고 유효성 검사
        bounds = standardize_bounds(bounds, x0, 'new')
        bounds = _validate_bounds(bounds, x0, meth)

        # meth가 'tnc', 'slsqp', 'l-bfgs-b'인 경우 경계의 하한과 상한이 같은 변수를 찾음
        if meth in {'tnc', 'slsqp', 'l-bfgs-b'}:
            i_fixed = (bounds.lb == bounds.ub)

        # 모든 변수의 경계가 같은 경우 최적화 결과를 반환
        if np.all(i_fixed):
            return _optimize_result_for_equal_bounds(
                fun, bounds, meth, args=args, constraints=constraints
            )

        # jac이 호출 가능한 함수가 아니거나 제약 조건의 자코비안이 호출 가능한 함수가 아닌 경우
        # fd_needed를 True로 설정
        fd_needed = (not callable(jac))
        for con in constraints:
            if not callable(con.get('jac', None)):
                fd_needed = True

        # i_fixed가 True이고 fd_needed가 True이거나 meth가 'tnc'인 경우
        # remove_vars를 True로 설정
        remove_vars = i_fixed.any() and (fd_needed or meth == 'tnc')
        # 일부 변수 제거
        if remove_vars:
            x_fixed = (bounds.lb)[i_fixed]
            x0 = x0[~i_fixed]
            bounds = _remove_from_bounds(bounds, i_fixed)
            fun = _remove_from_func(fun, i_fixed, x_fixed)
            if callable(callback):
                callback = _remove_from_func(callback, i_fixed, x_fixed)
            if callable(jac):
                jac = _remove_from_func(jac, i_fixed, x_fixed, remove=1)

            constraints = [con.copy() for con in constraints]
            for con in constraints:
                con['fun'] = _remove_from_func(con['fun'], i_fixed,
                                               x_fixed, min_dim=1,
                                               remove=0)
                if callable(con.get('jac', None)):
                    con['jac'] = _remove_from_func(con['jac'], i_fixed,
                                                   x_fixed, min_dim=2,
                                                   remove=1)
        # 경계 다시 표준화
        bounds = standardize_bounds(bounds, x0, meth)

    # callback 함수 래핑
    callback = _wrap_callback(callback, meth)

    # 최적화 방법에 따라 최적화 수행
    if meth == 'nelder-mead':
        res = _minimize_neldermead(fun, x0, args, callback, bounds=bounds,
                                   **options)
    elif meth == 'powell':
        res = _minimize_powell(fun, x0, args, callback, bounds, **options)
    elif meth == 'cg':
        res = _minimize_cg(fun, x0, args, jac, callback, **options)
    elif meth == 'bfgs':
        res = _minimize_bfgs(fun, x0, args, jac, callback, **options)
    elif meth == 'newton-cg':
        res = _minimize_newtoncg(fun, x0, args, jac, hess, hessp, callback,
                                 **options)
    elif meth == 'l-bfgs-b':
        res = _minimize_lbfgsb(fun, x0, args, jac, bounds,
                               callback=callback, **options)
    elif meth == 'tnc':
        res = _minimize_tnc(fun, x0, args, jac, bounds, callback=callback,
                            **options)
    elif meth == 'cobyla':
        res = _minimize_cobyla(fun, x0, args, constraints, callback=callback,
                               bounds=bounds, **options)
    elif meth == 'slsqp':
        res = _minimize_slsqp(fun, x0, args, jac, bounds,
                              constraints, callback=callback, **options)
    elif meth == 'trust-constr':
        res = _minimize_trustregion_constr(fun, x0, args, jac, hess, hessp,
                                           bounds, constraints,
                                           callback=callback, **options)
    elif meth == 'dogleg':
        res = _minimize_dogleg(fun, x0, args, jac, hess,
                               callback=callback, **options)
    elif meth == 'trust-ncg':
        res = _minimize_trust_ncg(fun, x0, args, jac, hess, hessp,
                                  callback=callback, **options)
    elif meth == 'trust-krylov':
        res = _minimize_trust_krylov(fun, x0, args, jac, hess, hessp,
                                     callback=callback, **options)
    elif meth == 'trust-exact':
        res = _minimize_trustregion_exact(fun, x0, args, jac, hess,
                                          callback=callback, **options)
    else:
        raise ValueError('Unknown solver %s' % method)

    # 제거된 변수 다시 추가
    if remove_vars:
        res.x = _add_to_array(res.X, i_fixed, x_fixed)
        res.jac = _add_to_array(res.jac, i_fixed, np.nan)
        if 'hess_inv' in res:
            res.hess_inv = None

    # callback 함수가 `StopIteration`을 발생시킨 경우
    if getattr(callback, 'stop_iteration', False):
        res.success = False
        res.status = 99
        res.message = '`callback` raised `StopIteration`.'

    return res
