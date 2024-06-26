import numpy as np


# 충분한 감소 조건과 곡률 조건을 만족하는 스텝 사이즈를 찾는 DCSRCH 클래스
class DCSRCH:
    # 초기화 메서드
    def __init__(self, phi, derphi, ftol, gtol, xtol, stpmin, stpmax):
        self.stage = None  # 최적화 단계
        self.ginit = None  # 초기 그레디언트 값
        self.gtest = None  # 그레디언트의 임계값
        self.gx = None  # 구간의 한쪽 끝점에서의 그레디언트 값
        self.gy = None  # 구간의 다른 쪽 끝점에서의 그레디언트 값
        self.finit = None  # 초기 함수 값
        self.fx = None  # 구간의 한쪽 끝점에서의 함수 값
        self.fy = None  # 구간의 다른 쪽 끝점에서의 함수 값
        self.stx = None  # 구간의 한쪽 끝점에서의 스텝 값
        self.sty = None  # 구간의 다른 쪽 끝점에서의 스텝 값
        self.stmin = None  # 스텝의 최소 값
        self.stmax = None  # 스텝의 최대 값
        self.width = None  # 구간의 현재 너비
        self.width1 = None  # 구간의 이전 너비

        self.ftol = ftol  # 함수 값의 허용 오차
        self.gtol = gtol  # 그레디언트의 허용 오차
        self.xtol = xtol  # 스텝의 허용 오차
        self.stpmin = stpmin  # 스텝의 최소 크기
        self.stpmax = stpmax  # 스텝의 최대 크기

        self.phi = phi  # 함수 값 계산 함수
        self.derphi = derphi  # 도함수 값 계산 함수

    # 인스턴스를 호출할 때 실행되는 메서드
    def __call__(self, alpha1, phi0=None, derphi0=None, maxiter=100):
        # 초기 함수 값, 도함수 값이 주어지지 않은 경우 0에서의 함수 값, 도함수 값 계산
        if phi0 is None:
            phi0 = self.phi(0.0)
        if derphi0 is None:
            derphi0 = self.derphi(0.0)

        # phi1, derphi1을 초기 함수 값, 도함수 값으로 초기화
        phi1 = phi0
        derphi1 = derphi0

        task = b'START'  # task를 'START'로 초기화
        for i in range(maxiter):
            # _iterate 메서드를 호출해 스텝 사이즈, 함수 값, 도함수 값, task를 업데이트
            stp, phi1, derphi1, task = self._iterate(
                alpha1, phi1, derphi1, task
            )

            if not np.isfinite(stp):  # stp가 유효한 값이 아닌 경우
                task = b'WARN'  # task를 'WARN'으로 설정
                stp = None  # stp를 None으로 설정
                break  # 반복문 종료

            if task[:2] == b'FG':  # task가 'FG'로 시작하는 경우
                alpha1 = stp  # alpha1을 stp로 설정
                phi1 = self.phi(stp)  # phi1을 stp에서의 함수 값으로 설정
                derphi1 = self.derphi(stp)  # derphi1을 stp에서의 도함수 값으로 설정
            else:  # 그 외의 경우
                break  # 반복문 종료
        # 모든 반복이 완료되면 stp를 None으로 설정하고 task를 경고 메시지로 설정
        else:
            stp = None
            task = b'WARNING: dcsrch did not converge within max iterations'

        # task가 에러 또는 경고 메시지인 경우 stp를 None으로 설정
        if task[:5] == b'ERROR' or task[:4] == b'WARN':
            stp = None

        return stp, phi1, phi0, task

    # 최적의 스텝 사이즈, 함수 값, 도함수 값, task를 찾아가는 메서드
    def _iterate(self, stp, f, g, task):
        p5 = 0.5
        p66 = 0.66
        xtrapl = 1.1
        xtrapu = 4.0

        # task가 'START'로 시작하는 경우
        if task[:5] == b'START':
            # stp가 stpmin보다 작은 경우 task를 에러 메시지로 설정
            if stp < self.stpmin:
                task = b'ERROR: STP .LT. STPMIN'
            # stp가 stpmax보다 큰 경우 task를 에러 메시지로 설정
            if stp > self.stpmax:
                task = b'ERROR: STP .GT. STPMAX'
            # g가 0보다 크거나 같은 경우 task를 에러 메시지로 설정
            if g >= 0:
                task = b'ERROR: INITIAL G .GE. ZERO'
            # ftol이 0보다 작은 경우 task를 에러 메시지로 설정
            if self.ftol < 0:
                task = b'ERROR: FTOL .LT. ZERO'
            # gtol이 0보다 작은 경우 task를 에러 메시지로 설정
            if self.gtol < 0:
                task = b'ERROR: GTOL .LT. ZERO'
            # xtol이 0보다 작은 경우 task를 에러 메시지로 설정
            if self.xtol < 0:
                task = b'ERROR: XTOL .LT. ZERO'
            # stpmin이 0보다 작은 경우 task를 에러 메시지로 설정
            if self.stpmin < 0:
                task = b'ERROR: STPMIN .LT. ZERO'
            # stpmax가 stpmin보다 작은 경우 task를 에러 메시지로 설정
            if self.stpmax < self.stpmin:
                task = b'ERROR: STPMAX .LT. STPMIN'

            # task가 ERROR로 시작하는 경우 stp, f, g, task를 반환
            if task[:5] == b'ERROR':
                return stp, f, g, task

            self.brackt = False  # brackt를 False로 설정
            self.stage = 1  # 최적화 단계를 1로 설정
            self.finit = f  # 초기 함수 값 설정
            self.ginit = g  # 초기 그레디언트 값 설정
            self.gtest = self.ftol * self.ginit  # 그레디언트의 임계값 설정
            self.width = self.stpmax - self.stpmin  # 구간의 너비 설정
            self.width1 = self.width / p5  # 구간의 이전 너비 설정

            self.stx = 0.0  # 구간의 한쪽 끝점에서의 스텝 값 초기화
            self.fx = self.finit  # 구간의 한쪽 끝점에서의 함수 값 초기화
            self.gx = self.ginit  # 구간의 한쪽 끝점에서의 그레디언트 값 초기화
            self.sty = 0.0  # 구간의 다른 쪽 끝점에서의 스텝 값 초기화
            self.fy = self.finit  # 구간의 다른 쪽 끝점에서의 함수 값 초기화
            self.gy = self.ginit  # 구간의 다른 쪽 끝점에서의 그레디언트 값 초기화
            self.stmin = 0  # 스텝의 최소 값 초기화
            self.stmax = stp + xtrapu * stp  # 스텝의 최대 값 초기화
            task = b'FG'  # task를 'FG'로 설정
            return stp, f, g, task

        # 현제 stp에서의 예상 함수 값
        ftest = self.finit + stp * self.gtest

        # 최적화 단계가 1인고, 함수 값이 예상 함수 값보다 작고, 그레디언트가 0보다 큰 경우 최적화 단계를 2로 설정
        if self.stage == 1 and f <= ftest and g >= 0:
            self.stage = 2

        # brackt가 True이고, stp가 stmin보다 작거나 stmax보다 큰 경우 task를 경고 메시지로 설정
        if self.brackt and (stp <= self.stmin or stp >= self.stmax):
            task = b'WARNING: ROUNDING ERRORS PREVENT PROGRESS'
        # brackt가 True이고, 스텝의 너비가 xtol * stmax보다 작은 경우 task를 경고 메시지로 설정
        if self.brackt and self.stamx - self.stmin <= self.xtol * self.stmax:
            task = b'WARNING: XTOL TEST SATISFIED'
        # stp가 stpmax이고, 함수 값이 예상 함수 값보다 작고, 그레디언트가 임계값보다 작은 경우
        # task를 경고 메시지로 설정
        if stp == self.stpmax and f <= ftest and g <= self.gtest:
            task = b'WARNING: STP = STPMAX'
        # stp가 stpmin이고, 함수 값이 예상 함수 값보다 크거나 그레디언트가 임계값보다 큰 경우
        # task를 경고 메시지로 설정
        if stp == self.stpmin and (f > ftest or g >= self.gtest):
            task = b'WARNING: STP = STPMIN'

        # 함수 값이 예상 함수 값보다 작고 그레디언트 절대값이 gtol * -ginit보다 작은 경우 task를 수렴 메시지로 설정
        if f <= ftest and abs(g) <= self.gtol * -self.ginit:
            task = b'CONVERGENCE'

        # task가 경고 또는 수렴 메시지인 경우 stp, f, g, task를 반환
        if task[:4] == b'WARN' or task[:4] == b'CONV':
            return stp, f, g, task

        # 최적화 단계가 1이고 함수 값이 한쪽 끝점에서의 함수 값보다 작거나 같고 다른 쪽 끝점에서의 함수 값보다 큰 경우
        if self.stage == 1 and f <= self.fx and f > ftest:
            fm = f - stp * self.gtest  # 현재 스탭에서의 함수의 조정 값
            fxm = self.fx - self.stx * self.gtest  # 한쪽 끝점에서의 함수의 조정 값
            fym = self.fy - self.sty * self.gtest  # 다른 쪽 끝점에서의 함수의 조정 값
            gm = g - self.gtest  # 현재 그레디언트의 조정 값
            gxm = self.gx - self.gtest  # 한쪽 끝점에서의 그레디언트의 조정 값
            gym = self.gy - self.gtest  # 다른 쪽 끝점에서의 그레디언트의 조정 값

            # 특정 종류의 부동 소수점 오류를 무시하도록 설정
            with np.errstate(invalid='ignore', over='ignore'):
                # dcstep 함수를 이용해 구간과 스텝 사이즈를 업데이트
                tup = dcstep(
                    self.stx,
                    fxm,
                    gxm,
                    self.sty,
                    fym,
                    gym,
                    stp,
                    fm,
                    gm,
                    self.brackt,
                    self.stmin,
                    self.stmax,
                )
                self.stx, fxm, gxm, self.sty, fym, gym, stp, self.brackt = tup

            self.fx = fxm + self.stx * self.gtest  # 한쪽 끝점에서의 함수 값 업데이트
            self.fy = fym + self.sty * self.gtest  # 다른 쪽 끝점에서의 함수 값 업데이트
            self.gx = gym + self.gtest  # 한쪽 끝점에서의 그레디언트 값 업데이트
            self.gy = gym + self.gtest  # 다른 쪽 끝점에서의 그레디언트 값 업데이트

        # 그 외의 경우
        else:
            # 특정 종류의 부동 소수점 오류를 무시하도록 설정
            with np.errstate(invalid='ignore', over='ignore'):
                # dcstep 함수를 이용해 구간과 스텝 사이즈를 업데이트
                tup = dcstep(
                    self.stx,
                    self.fx,
                    self.gx,
                    self.sty,
                    self.fy,
                    self.gy,
                    stp,
                    f,
                    g,
                    self.brackt,
                    self.stmin,
                    self.stmax,
                )
            (
                self.stx,
                self.fx,
                self.gx,
                self.sty,
                self.fy,
                self.gy,
                stp,
                self.brackt,
            ) = tup

        # brackt가 True인 경우
        if self.brackt:
            # 스텝 구간의 너비가 이전 너비의 66%보다 큰 경우
            if abs(self.sty - self.stx) >= p66 * self.width1:
                # 스텝을 스텝 구간의 중점으로 설정
                stp = self.stx + p5 * (self.sty - self.stx)
            self.width1 = self.width  # 이전 너비를 현재 너비로 설정
            self.width = abs(self.sty - self.stx)  # 현재 너비 업데이트

        # brackt가 True인 경우
        if self.brackt:
            # 스텝 구간의 한쪽 끝점과 다른 쪽 끝점을 stmin, stmax로 설정
            self.stmin = min(self.stx, self.sty)
            self.stmax = max(self.stx, self.sty)
        # brackt가 False인 경우
        else:
            # stmin을 stp에서 1.1배 확장한 값으로, stmax를 stp에서 4배 확장한 값으로 설정
            self.stmin = stp + xtrapl * (stp - self.stx)
            self.stmax = stp + xtrapu * (stp - self.stx)

        # stp를 stpmin, stpmax 사이의 값으로 제한
        stp = np.clip(stp, self.stpmin, self.stpmax)

        if (  # self.brackt가 True이고 stp가 stmin보다 작거나 stmax보다 큰 경우
            self.brackt
            and (stp <= self.stmin or stp >= self.stmax)
            or (
                self.brackt
                and self.stmax - self.stmin <= self.xtol * self.stmax
            )  # 또는 self.brackt가 True이고 스텝의 너비가 xtol * stmax보다 작은 경우
        ):
            stp = self.stx  # stp를 스텝 구간의 한쪽 끝점으로 설정

        task = b'FG'  # task를 'FG'로 설정
        return stp, f, g, task


# 충분한 감소 조건과 곡률 조건을 만족하는 스텝 사이즈를 찾아가는 dcstep 함수
def dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    sgn_dp = np.sign(dp)  # dp의 부호
    sgn_dx = np.sign(dx)  # dx의 부호

    sgnd = sgn_dp * sgn_dx  # dp의 부호와 dx의 부호의 곱

    if fp > fx:  # 함수값이 구간의 한쪽 끝점에서의 함수값보다 큰 경우
        # theta, s, gamma, p, q, r은 보간에 사용되는 값
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp < stx:
            gamma *= -1
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        # stpc는 보간된 스텝 값
        stpc = stx + r * (stp - stx)
        # stpq는 다른 방법으로 보간된 스텝 값
        stpq = stx + ((dx / ((fx - fp) /
                             (stp - stx) + dx)) / 2.0) * (stp - stx)
        # stpc가 stpq보다 스텝 구간의 한쪽 끝점에 더 가까운 경우
        if abs(stpc - stx) <= abs(stpq - stx):
            stpf = stpc  # stpf를 stpc로 설정
        # 그 외의 경우
        else:
            # stpc와 stpq의 중점을 stpf로 설정
            stpf = stpc + (stpq - stpc) / 2.0
        brackt = True  # brackt를 True로 설정
    # 그레디언트의 부호가 바뀌는 경우 stpf 계산
    elif sgnd < 0.0:
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp > stx:
            gamma *= -1
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True
    # 현재 스텝에서의 그레디언트 절대값이 이전 스텝에서의 절대값보다 작은 경우 stpf 계산
    elif abs(dp) < abs(dx):
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))

        gamma = s * np.sqrt(max(0, (theta / s) ** 2 - (dx / s) * (dp / s)))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        if r < 0 and gamma != 0:
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp / (dp - dx)) * (stx - stp)

        if brackt:
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq

            if stp > stx:
                stpf = min(stp + 0.66 * (sty - stp), stpf)
            else:
                stpf = max(stp + 0.66 * (sty - stp), stpf)
        else:
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = np.clip(stpf, stpmin, stpmax)

    # 그 외의 경우 stpf 계산
    else:
        if brackt:
            theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
            s = max(abs(theta), abs(dy), abs(dp))
            gamma = s * np.sqrt((theta / s) ** 2 - (dy / s) * (dp / s))
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            stpc = stp + r * (sty - stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    # 함수 값이 구간의 한쪽 끝점에서의 함수 값보다 큰 경우
    if fp > fx:
        sty = stp  # 다른 쪽 끝점에서의 스텝 값을 현재 스텝 값으로 설정
        fx = fp  # 한쪽 끝점에서의 함수 값을 현재 함수 값으로 설정
        dy = dp  # 다른 쪽 끝점에서의 그레디언트 값을 현재 그레디언트 값으로 설정
    else:  # 그 외의 경우
        if sgnd < 0:  # 그레디언트의 부호가 바뀌는 경우
            sty = stx  # 다른 쪽 끝점에서의 스텝 값을 한쪽 끝점에서의 스텝 값으로 설정
            fy = fx  # 다른 쪽 끝점에서의 함수 값을 한쪽 끝점에서의 함수 값으로 설정
            dy = dx  # 다른 쪽 끝점에서의 그레디언트 값을 한쪽 끝점에서의 그레디언트 값으로 설정
        stx = stp  # 한쪽 끝점에서의 스텝 값을 현재 스텝 값으로 설정
        fx = fp  # 한쪽 끝점에서의 함수 값을 현재 함수 값으로 설정
        dx = dp  # 한쪽 끝점에서의 그레디언트 값을 현재 그레디언트 값으로 설정

    stp = stpf  # 스텝 값을 보간된 스텝 값으로 설정

    return stx, fx, dx, sty, fy, dy, stp, brackt
