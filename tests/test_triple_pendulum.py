import numpy as np
from casadi import MX, vertcat, sin, cos, Function


class Parameters:
    def __init__(self):
        self.m1 = 0.4
        self.m2 = 0.4
        self.m3 = 0.4
        self.l1 = 0.8
        self.l2 = 0.8
        self.l3 = 0.8
        self.g = 9.81


class DirectDynamics:
    def __init__(self, params):
        """ Define the triple pendulum dynamics model. """

        self.x = MX.sym("x", 6)  # state = (position, velocity)
        self.x_dot = MX.sym("x_dot", 6)  # x_dot = f(x, u)
        self.u = MX.sym("u", 3)  # torque
        self.p = MX.sym("p", 1)

        # dynamics --> x_dot = f(x, u)
        f_expl = vertcat(
            self.x[3],
            self.x[4],
            self.x[5],
            (
                -params.g
                * params.l1
                * params.l2
                * params.l2
                * params.m1
                * params.m3
                * sin(-2 * self.x[2] + 2 * self.x[1] + self.x[0])
                - params.g
                * params.l1
                * params.l2
                * params.l2
                * params.m1
                * params.m3
                * sin(2 * self.x[2] - 2 * self.x[1] + self.x[0])
                + 2
                * self.u[0]
                * params.l2
                * params.l2
                * params.m3
                * cos(-2 * self.x[2] + 2 * self.x[1])
                + 2
                * self.x[3] ** 2
                * params.l1 ** 2
                * params.l2
                * params.l2
                * params.m2
                * (params.m2 + params.m3)
                * sin(-2 * self.x[1] + 2 * self.x[0])
                - 2
                * self.u[2]
                * params.l1
                * params.l2
                * (params.m2 + params.m3)
                * cos(-2 * self.x[1] + self.x[0] + self.x[2])
                - 2
                * self.u[1]
                * params.l1
                * params.l2
                * params.m3
                * cos(-2 * self.x[2] + self.x[1] + self.x[0])
                + 2
                * params.l1
                * params.l2
                * params.l2 ** 2
                * params.m2
                * params.m3
                * self.x[5] ** 2
                * sin(-2 * self.x[1] + self.x[0] + self.x[2])
                + 2
                * self.u[2]
                * params.l1
                * params.l2
                * (params.m2 + params.m3)
                * cos(self.x[0] - self.x[2])
                + 2
                * (
                    self.u[1]
                    * params.l1
                    * (params.m3 + 2 * params.m2)
                    * cos(-self.x[1] + self.x[0])
                    + (
                        params.g
                        * params.l1
                        * params.m2
                        * (params.m2 + params.m3)
                        * sin(-2 * self.x[1] + self.x[0])
                        + 2
                        * self.x[4] ** 2
                        * params.l1
                        * params.l2
                        * params.m2
                        * (params.m2 + params.m3)
                        * sin(-self.x[1] + self.x[0])
                        + params.m3
                        * self.x[5] ** 2
                        * sin(self.x[0] - self.x[2])
                        * params.l1
                        * params.l2
                        * params.m2
                        + params.g
                        * params.l1
                        * (
                            params.m2 ** 2
                            + (params.m3 + 2 * params.m1) * params.m2
                            + params.m1 * params.m3
                        )
                        * sin(self.x[0])
                        - self.u[0] * (params.m3 + 2 * params.m2)
                    )
                    * params.l2
                )
                * params.l2
            )
            / params.l1 ** 2
            / params.l2
            / (
                params.m2
                * (params.m2 + params.m3)
                * cos(-2 * self.x[1] + 2 * self.x[0])
                + params.m1 * params.m3 * cos(-2 * self.x[2] + 2 * self.x[1])
                - params.m2 ** 2
                + (-params.m3 - 2 * params.m1) * params.m2
                - params.m1 * params.m3
            )
            / params.l2
            / 2,
            (
                -2
                * self.u[2]
                * params.l1
                * params.l2
                * (params.m2 + params.m3)
                * cos(2 * self.x[0] - self.x[2] - self.x[1])
                - 2
                * params.l1
                * params.l2
                * params.l2 ** 2
                * params.m2
                * params.m3
                * self.x[5] ** 2
                * sin(2 * self.x[0] - self.x[2] - self.x[1])
                + params.g
                * params.l1
                * params.l2
                * params.l2
                * params.m1
                * params.m3
                * sin(self.x[1] + 2 * self.x[0] - 2 * self.x[2])
                - params.g
                * params.l1
                * params.l2
                * (
                    (params.m1 + 2 * params.m2) * params.m3
                    + 2 * params.m2 * (params.m1 + params.m2)
                )
                * params.l2
                * sin(-self.x[1] + 2 * self.x[0])
                - 2
                * self.x[4] ** 2
                * params.l1
                * params.l2 ** 2
                * params.l2
                * params.m2
                * (params.m2 + params.m3)
                * sin(-2 * self.x[1] + 2 * self.x[0])
                + 2
                * self.u[1]
                * params.l1
                * params.l2
                * params.m3
                * cos(-2 * self.x[2] + 2 * self.x[0])
                + 2
                * params.l1
                * params.l2 ** 2
                * params.l2
                * params.m1
                * params.m3
                * self.x[4] ** 2
                * sin(-2 * self.x[2] + 2 * self.x[1])
                - 2
                * self.u[0]
                * params.l2
                * params.l2
                * params.m3
                * cos(-2 * self.x[2] + self.x[1] + self.x[0])
                + 2
                * params.l1 ** 2
                * params.l2
                * params.l2
                * params.m1
                * params.m3
                * self.x[3] ** 2
                * sin(-2 * self.x[2] + self.x[1] + self.x[0])
                - 2
                * params.l1 ** 2
                * params.l2
                * self.x[3] ** 2
                * (
                    (params.m1 + 2 * params.m2) * params.m3
                    + 2 * params.m2 * (params.m1 + params.m2)
                )
                * params.l2
                * sin(-self.x[1] + self.x[0])
                + 2
                * self.u[2]
                * params.l1
                * params.l2
                * (params.m3 + 2 * params.m1 + params.m2)
                * cos(-self.x[2] + self.x[1])
                + (
                    2
                    * self.u[0]
                    * params.l2
                    * (params.m3 + 2 * params.m2)
                    * cos(-self.x[1] + self.x[0])
                    + params.l1
                    * (
                        4
                        * self.x[5] ** 2
                        * params.m3
                        * params.l2
                        * (params.m1 + params.m2 / 2)
                        * params.l2
                        * sin(-self.x[2] + self.x[1])
                        + params.g
                        * params.m3
                        * params.l2
                        * params.m1
                        * sin(-2 * self.x[2] + self.x[1])
                        + params.g
                        * (
                            (params.m1 + 2 * params.m2) * params.m3
                            + 2 * params.m2 * (params.m1 + params.m2)
                        )
                        * params.l2
                        * sin(self.x[1])
                        - 2 * self.u[1] * (params.m3 + 2 * params.m1 + 2 * params.m2)
                    )
                )
                * params.l2
            )
            / (
                params.m2
                * (params.m2 + params.m3)
                * cos(-2 * self.x[1] + 2 * self.x[0])
                + params.m1 * params.m3 * cos(-2 * self.x[2] + 2 * self.x[1])
                + (-params.m1 - params.m2) * params.m3
                - 2 * params.m1 * params.m2
                - params.m2 ** 2
            )
            / params.l1
            / params.l2
            / params.l2 ** 2
            / 2,
            (
                -2
                * params.m3
                * self.u[1]
                * params.l1
                * params.l2
                * (params.m2 + params.m3)
                * cos(2 * self.x[0] - self.x[2] - self.x[1])
                + params.g
                * params.m3
                * params.l1
                * params.l2
                * params.l2
                * params.m1
                * (params.m2 + params.m3)
                * sin(2 * self.x[0] + self.x[2] - 2 * self.x[1])
                + 2
                * self.u[2]
                * params.l1
                * params.l2
                * (params.m2 + params.m3) ** 2
                * cos(-2 * self.x[1] + 2 * self.x[0])
                - params.g
                * params.m3
                * params.l1
                * params.l2
                * params.l2
                * params.m1
                * (params.m2 + params.m3)
                * sin(2 * self.x[0] - self.x[2])
                - params.g
                * params.m3
                * params.l1
                * params.l2
                * params.l2
                * params.m1
                * (params.m2 + params.m3)
                * sin(-self.x[2] + 2 * self.x[1])
                - 2
                * params.l1
                * params.l2
                * params.l2 ** 2
                * params.m1
                * params.m3 ** 2
                * self.x[5] ** 2
                * sin(-2 * self.x[2] + 2 * self.x[1])
                - 2
                * self.u[0]
                * params.l2
                * params.l2
                * params.m3
                * (params.m2 + params.m3)
                * cos(-2 * self.x[1] + self.x[0] + self.x[2])
                + 2
                * params.m3
                * self.x[3] ** 2
                * params.l1 ** 2
                * params.l2
                * params.l2
                * params.m1
                * (params.m2 + params.m3)
                * sin(-2 * self.x[1] + self.x[0] + self.x[2])
                + 2
                * params.m3
                * self.u[1]
                * params.l1
                * params.l2
                * (params.m3 + 2 * params.m1 + params.m2)
                * cos(-self.x[2] + self.x[1])
                + (params.m2 + params.m3)
                * (
                    2 * self.u[0] * params.l2 * params.m3 * cos(self.x[0] - self.x[2])
                    + params.l1
                    * (
                        -2
                        * params.m3
                        * self.x[3] ** 2
                        * params.l1
                        * params.l2
                        * params.m1
                        * sin(self.x[0] - self.x[2])
                        - 4
                        * params.m3
                        * self.x[4] ** 2
                        * sin(-self.x[2] + self.x[1])
                        * params.l2
                        * params.l2
                        * params.m1
                        + params.g * params.m3 * sin(self.x[2]) * params.l2 * params.m1
                        - 2 * self.u[2] * (params.m3 + 2 * params.m1 + params.m2)
                    )
                )
                * params.l2
            )
            / params.m3
            / (
                params.m2
                * (params.m2 + params.m3)
                * cos(-2 * self.x[1] + 2 * self.x[0])
                + params.m1 * params.m3 * cos(-2 * self.x[2] + 2 * self.x[1])
                + (-params.m1 - params.m2) * params.m3
                - 2 * params.m1 * params.m2
                - params.m2 ** 2
            )
            / params.l1
            / params.l2 ** 2
            / params.l2
            / 2,
        )

        self.f_expl_fun = Function("f_expl", [self.x, self.u], [f_expl])


class InverseDynamics:
    def __init__(self, params):

        self.x = MX.sym("x", 6)  # state = (position, velocity)
        self.a = MX.sym("u", 3)  # acceleration

        # dynamics --> u = f(x, a)
        f_inv = vertcat(
            params.l1
            * (
                self.a[0] * params.l1 * params.m1
                + self.a[0] * params.l1 * params.m2
                + self.a[0] * params.l1 * params.m3
                + self.a[1]
                * params.l2
                * (params.m2 + params.m3)
                * cos(self.x[0] - self.x[1])
                + self.a[2] * params.l2 * params.m3 * cos(self.x[0] - self.x[2])
                + params.g * params.m1 * sin(self.x[0])
                + params.g * params.m2 * sin(self.x[0])
                + params.g * params.m3 * sin(self.x[0])
                + params.l2 * params.m2 * self.x[4] ** 2 * sin(self.x[0] - self.x[1])
                + params.l2 * params.m3 * self.x[4] ** 2 * sin(self.x[0] - self.x[1])
                + params.l2 * params.m3 * self.x[5] ** 2 * sin(self.x[0] - self.x[2])
            ),
            params.l2
            * (
                self.a[1] * params.l2 * params.m2
                + self.a[1] * params.l2 * params.m3
                + self.a[0]
                * params.l1
                * (params.m2 + params.m3)
                * cos(self.x[0] - self.x[1])
                + self.a[2] * params.l2 * params.m3 * cos(self.x[1] - self.x[2])
                - params.l1 * params.m2 * self.x[3] ** 2 * sin(self.x[0] - self.x[1])
                - params.l1 * params.m3 * self.x[3] ** 2 * sin(self.x[0] - self.x[1])
                + params.g * params.m2 * sin(self.x[1])
                + params.g * params.m3 * sin(self.x[1])
                + params.l2 * params.m3 * self.x[5] ** 2 * sin(self.x[1] - self.x[2])
            ),
            params.l2
            * params.m3
            * (
                self.a[2] * params.l2
                + self.a[0] * params.l1 * cos(self.x[0] - self.x[2])
                + self.a[1] * params.l2 * cos(self.x[1] - self.x[2])
                - params.l1 * self.x[3] ** 2 * sin(self.x[0] - self.x[2])
                - params.l2 * self.x[4] ** 2 * sin(self.x[1] - self.x[2])
                + params.g * sin(self.x[2])
            ),
        )

        self.f_inv_fun = Function("f_inv", [self.x, self.a], [f_inv])


if __name__ == "__main__":
    params = Parameters()
    direct_dynamics = DirectDynamics(params)
    inverse_dynamics = InverseDynamics(params)

    x = np.ones(6)
    u = np.ones(3) * 2
    a = direct_dynamics.f_expl_fun(x, u)
    print(a)
    u_ = inverse_dynamics.f_inv_fun(x, a[3:])
    print(u_)       # OK !
