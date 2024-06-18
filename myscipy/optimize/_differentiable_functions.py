import numpy as np
import scipy.sparse as sps
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize._hessian_update_strategy import HessianUpdateStrategy
from scipy.sparse.linalg import LinearOperator
from scipy._lib._array_api import atleast_nd, array_namespace

FD_METHODS = ('2-point', '3-point', 'cs')  # 유한 차분 방법의 종류


# 스칼라 최소화를 위한 ScalarFunction 클래스
class ScalarFunction:
    # 초기화 메서드
    def __init__(self, fun, x0, args, grad, hess, finite_diff_rel_step,
                 finite_diff_bounds, epsilon=None):
        # grad가 호출 가능한 함수가 아니거나 FD_METHODS에 포함되지 않은 경우 에러 발생
        if not callable(grad) and grad not in FD_METHODS:
            raise ValueError(
                f'`grad` must be either callable or one of {FD_METHODS}.'
            )

        # hess가 호출 가능한 함수가 아니거나 FD_METHODS에 포함되지 않은 경우 에러 발생
        if not (callable(hess) or hess in FD_METHODS
                or isinstance(hess, HessianUpdateStrategy)):
            raise ValueError(
                f'`hess` must be either callable, HessianUpdateStrategy'
                f' or one of {FD_METHODS}.'
            )

        # grad가 FD_METHODS에 포함되어 있고 hess가 FD_METHODS에 포함되어 있으면 에러 발생
        if grad in FD_METHODS and hess in FD_METHODS:
            raise ValueError('Whenever the gradient is estimated via '
                             'finite-differences, we require the Hessian '
                             'to be estimated using one of the '
                             'quasi-Newton strategies.')

        # x0의 배열 네임스페이스를 확인하고 이를 xp, self.xp에 저장
        self.xp = xp = array_namespace(x0)
        # x0를 1차원 배열로 변환하고 이를 _x에 저장
        _x = atleast_nd(x0, ndim=1, xp=xp)
        # _x의 데이터 타입이 실수인 경우 _dtype을 _x의 데이터 타입으로 설정하고 아니면 float64로 설정
        _dtype = xp.float64
        if xp.isdtype(_x.dtype, 'real floating'):
            _dtype = _x.dtype

        self.x = xp.astype(_x, _dtype)  # _x를 _dtype으로 변환하고 이를 self.x에 저장
        self.x_dtype = _dtype  # _dtype을 self.x_dtype에 저장
        self.n = self.x.size  # self.x의 크기를 self.n에 저장
        self.nfev = 0  # 함수 계산 횟수를 0으로 초기화
        self.ngev = 0  # 그레디언트 계산 횟수를 0으로 초기화
        self.nhev = 0  # 헤세 계산 횟수를 0으로 초기화
        self.f_updated = False  # 함수 값이 업데이트 되었는지 여부를 False로 초기화
        self.g_updated = False  # 그레디언트가 업데이트 되었는지 여부를 False로 초기화
        self.H_updated = False  # 헤세가 업데이트 되었는지 여부를 False로 초기화

        self._lowest_x = None  # 최저 함수 값의 x를 저장할 변수를 None으로 초기화
        self._lowest_f = np.inf  # 최저 함수 값을 무한대로 초기화

        finite_diff_options = {}  # 유한 차분 방법 옵션을 저장할 딕셔너리 생성
        # grad가 FD_METHODS에 포함되어 있을 경우 finite_diff_options에
        # method, rel_step, abs_step, bounds 키를 추가
        if grad in FD_METHODS:
            finite_diff_options['method'] = grad
            finite_diff_options['rel_step'] = finite_diff_rel_step
            finite_diff_options['abs_step'] = epsilon
            finite_diff_options['bounds'] = finite_diff_bounds
        # hess가 FD_METHODS에 포함되어 있을 경우 finite_diff_options에
        # method, rel_step, abs_step, as_linear_operator 키를 추가
        if hess in FD_METHODS:
            finite_diff_options['method'] = hess
            finite_diff_options['rel_step'] = finite_diff_rel_step
            finite_diff_options['abs_step'] = epsilon
            finite_diff_options['as_linear_operator'] = True

        # fun 함수를 래핑한 내장 함수
        def fun_wrapped(x):
            self.nfev += 1  # 함수 계산 횟수 1 증가
            fx = fun(np.copy(x), *args)  # x를 복사하여 fun 함수에 전달한 후 결과를 fx에 저장
            # fx가 스칼라가 아닌 경우 스칼라 값으로 변환하려 시도하고 변환할 수 없으면 에러 발생
            if not np.isscalar(fx):
                try:
                    fx = np.asarray(fx).item()
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        'The user-provided objective function '
                        'must return a scalar value.'
                    ) from e

            # fx가 최저 함수 값보다 작으면 최저 함수 값과 그 때의 x를 업데이트
            if fx < self._lowest_f:
                self._lowest_x = x
                self._lowest_f = fx

            return fx

        # self.x에서의 함수 값을 계산하고 이를 self.f에 저장하는 내장 함수
        def update_fun():
            self.f = fun_wrapped(self.x)

        # update_fun 함수를 self._update_fun_impl에 저장
        self._update_fun_impl = update_fun
        # 함수 값이 업데이트 되지 않았다면 _update_fun 함수를 호출하여 self.f를 업데이트
        self._update_fun()

        # grad가 호출 가능한 경우
        if callable(grad):
            # grad 함수를 래핑한 내장 함수
            def grad_wrapped(x):
                self.ngev += 1  # 그레디언트 계산 횟수 1 증가
                # x를 복사하여 grad 함수에 전달한 후 결과를 반환
                return np.atleast_1d(grad(np.copy(x), *args))

            # self.x에서의 그레디언트를 계산하고 이를 self.g에 저장하는 내장 함수
            def update_grad():
                self.g = grad_wrapped(self.x)

        # grad가 유한 차분 방법 중 하나인 경우
        elif grad in FD_METHODS:
            # 유한 차분 방법을 이용해 그레디언트를 계산하고 이를 self.g에 저장하는 내장 함수
            def update_grad():
                # 함수 값이 업데이트 되지 않았다면 _update_fun 함수를 호출하여 self.f를 업데이트
                self._update_fun()
                self.ngev += 1  # 그레디언트 계산 횟수 1 증가
                # approx_derivative 함수를 이용해 그레디언트를 계산하고 이를 self.g에 저장
                self.g = approx_derivative(fun_wrapped, self.x, f0=self.f,
                                           **finite_diff_options)

        # update_grad 함수를 self._update_grad_impl에 저장
        self._update_grad_impl = update_grad
        # 그레디언트가 업데이트 되지 않았다면 _update_grad 함수를 호출하여 self.g를 업데이트
        self._update_grad()

        # hess가 호출 가능한 경우
        if callable(hess):
            # x0를 복사하여 hess 함수에 전달한 후 결과를 self.H에 저장
            self.H = hess(np.copy(x0), *args)
            # 헤세가 업데이트 되었음을 나타내는 self.H_updated를 True로 설정
            self.H_updated = True
            self.nhev += 1  # 헤세 계산 횟수 1 증가

            # 헤세 행렬이 희소 행렬인 경우
            if sps.issparse(self.H):
                # hess 함수를 래핑한 내장 함수
                def hess_wrapped(x):
                    self.nhev += 1  # 헤세 계산 횟수 1 증가
                    # x를 복사하여 hess 함수에 전달한 후 결과를 CSR 형식의 희소 행렬로 변환하여 반환
                    return sps.csr_matrix(hess(np.copy(x), *args))
                # self.H를 CSR 형식의 희소 행렬로 변환
                self.H = sps.csr_matrix(self.H)

            # 헤세 행렬이 선형 연산자인 경우
            elif isinstance(self.H, LinearOperator):
                # hess 함수를 래핑한 내장 함수
                def hess_wrapped(x):
                    self.nhev += 1  # 헤세 계산 횟수 1 증가
                    # x를 복사하여 hess 함수에 전달한 후 결과를 반환
                    return hess(np.copy(x), *args)

            # 그 외의 경우
            else:
                # hess 함수를 래핑한 내장 함수
                def hess_wrapped(x):
                    self.nhev += 1  # 헤세 계산 횟수 1 증가
                    # x를 복사하여 hess 함수에 전달한 후 결과를 2차원 배열로 변환하여 반환
                    return np.atleast_2d(np.asarray(hess(np.copy(x), *args)))
                # self.H를 2차원 배열로 변환
                self.H = np.atleast_2d(np.asarray(self.H))

            # self.x에서의 헤세 행렬을 계산하고 이를 self.H에 저장하는 내장 함수
            def update_hess():
                self.H = hess_wrapped(self.x)

        # hess가 유한 차분 방법 중 하나인 경우
        elif hess in FD_METHODS:
            # 유한 차분 방법을 이용해 헤세 행렬을 계산하고 이를 self.H에 저장하는 내장 함수
            def update_hess():
                # 그레디언트가 업데이트 되지 않았다면 _update_grad 함수를 호출하여 self.g를 업데이트
                self._update_grad()
                # approx_derivative 함수를 이용해 헤세 행렬을 계산하고 이를 self.H에 저장
                self.H = approx_derivative(grad_wrapped, self.x, f0=self.g,
                                           **finite_diff_options)
                return self.H

            update_hess()  # update_hess 함수를 호출하여 self.H를 업데이트
            # 헤세가 업데이트 되었음을 나타내는 self.H_updated를 True로 설정
            self.H_updated = True
        # hess가 HessianUpdateStrategy 인스턴스인 경우
        elif isinstance(hess, HessianUpdateStrategy):
            self.H = hess  # hess를 self.H에 저장
            self.H.initialize(self.n, 'hess')  # self.H를 초기화
            # 헤세가 업데이트 되었음을 나타내는 self.H_updated를 True로 설정
            self.H_updated = True
            self.x_prev = None  # 이전 self.x를 저장할 변수를 None으로 초기화
            self.g_prev = None  # 이전 self.g를 저장할 변수를 None으로 초기화

            # 헤세 행렬을 업데이트하는 내장 함수
            def update_hess():
                # 그레디언트가 업데이트 되지 않았다면 _update_grad 함수를 호출하여 self.g를 업데이트
                self._update_grad()
                # 이전 x와 g를 이용해 헤세 행렬을 업데이트
                self.H.update(self.x - self.x_prev, self.g - self.g_prev)

        # update_hess 함수를 self._update_hess_impl에 저장
        self._update_hess_impl = update_hess

        # hess가 HessianUpdateStrategy인 경우
        if isinstance(hess, HessianUpdateStrategy):
            # self.x를 업데이트하는 내장 함수
            def update_x(x):
                # 그레디언트가 업데이트 되지 않았다면 _update_grad 함수를 호출하여 self.g를 업데이트
                self._update_grad()
                self.x_prev = self.x  # 현재 self.x를 이전 x로 설정
                self.g_prev = self.g  # 현재 self.g를 이전 g로 설정

                # x를 1차원 배열로 변환하고 이를 _x에 저장
                _x = atleast_nd(x, ndim=1, xp=self.xp)
                # _x의 데이터 타입을 self.x_dtype으로 설정하고 이를 self.x에 저장
                self.x = self.xp.astype(_x, self.x_dtype)
                # 함수 값이 업데이트 되었음을 나타내는 self.f_updated를 False로 설정
                self.f_updated = False
                # 그레디언트가 업데이트 되었음을 나타내는 self.g_updated를 False로 설정
                self.g_updated = False
                # 헤세가 업데이트 되었음을 나타내는 self.H_updated를 False로 설정
                self.H_updated = False
                # 헤세가 업데이트 되지 않았다면 _update_hess 함수를 호출하여 self.H를 업데이트
                self._update_hess()
        # hess가 HessianUpdateStrategy가 아닌 경우
        else:
            # self.x를 업데이트하는 내장 함수
            def update_x(x):
                # x를 1차원 배열로 변환하고 이를 _x에 저장
                _x = atleast_nd(x, ndim=1, xp=self.xp)
                # _x의 데이터 타입을 self.x_dtype으로 설정하고 이를 self.x에 저장
                self.x = self.xp.astype(_x, self.x_dtype)
                # 함수 값이 업데이트 되었음을 나타내는 self.f_updated를 False로 설정
                self.f_updated = False
                # 그레디언트가 업데이트 되었음을 나타내는 self.g_updated를 False로 설정
                self.g_updated = False
                # 헤세가 업데이트 되었음을 나타내는 self.H_updated를 False로 설정
                self.H_updated = False
        # update_x 함수를 self._update_x_impl에 저장
        self._update_x_impl = update_x

    # 함수 값이 업데이트 되지 않았다면 _update_fun_impl를 호출하여 self.f를 업데이트
    def _update_fun(self):
        if not self.f_updated:
            self._update_fun_impl()
            self.f_updated = True

    # 그레디언트가 업데이트 되지 않았다면 _update_grad_impl를 호출하여 self.g를 업데이트
    def _update_grad(self):
        if not self.g_updated:
            self._update_grad_impl()
            self.g_updated = True

    # 헤세가 업데이트 되지 않았다면 _update_hess_impl를 호출하여 self.H를 업데이트
    def _update_hess(self):
        if not self.H_updated:
            self._update_hess_impl()
            self.H_updated = True

    # x에 대한 함수 값을 반환하는 메서드
    def fun(self, x):
        # x가 self.x와 다른 경우 _update_x_impl를 호출하여 self.x를 업데이트
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        # 함수 값이 업데이트 되지 않았다면 _update_fun 메서드를 호출하여 self.f를 업데이트
        self._update_fun()
        return self.f

    # x에 대한 그레디언트를 반환하는 메서드
    def grad(self, x):
        # x가 self.x와 다른 경우 _update_x_impl를 호출하여 self.x를 업데이트
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        # 그레디언트가 업데이트 되지 않았다면 _update_grad 메서드를 호출하여 self.g를 업데이트
        self._update_grad()
        return self.g

    # x에 대한 헤세 행렬을 반환하는 메서드
    def hess(self, x):
        # x가 self.x와 다른 경우 _update_x_impl를 호출하여 self.x를 업데이트
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        # 헤세가 업데이트 되지 않았다면 _update_hess 메서드를 호출하여 self.H를 업데이트
        self._update_hess()
        return self.H

    # x에 대한 함수 값과 그레디언트를 반환하는 메서드
    def fun_and_grad(self, x):
        # x가 self.x와 다른 경우 _update_x_impl를 호출하여 self.x를 업데이트
        if not np.array_equal(self, x):
            self._update_x_impl(x)
        # 함수 값이 업데이트 되지 않았다면 _update_fun 메서드를 호출하여 self.f를 업데이트
        self._update_fun()
        # 그레디언트가 업데이트 되지 않았다면 _update_grad 메서드를 호출하여 self.g를 업데이트
        self._update_grad()
        return self.f, self.g
