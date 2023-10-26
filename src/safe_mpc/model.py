from casadi import SX, vertcat, cos, sin
from .abstract import AbstractModel


class DoublePendulumModel(AbstractModel):
    def __init__(self, params):
        super().__init__(params)

    def addDynamicsModel(self, params):
        """ Define the double pendulum dynamics model. """
        self.model.name = "double_pendulum"

        self.x = SX.sym("x", 4)
        self.x_dot = SX.sym("x_dot", 4)
        self.u = SX.sym("u", 2)
        self.p = SX.sym("p", 1)

        # Dynamics
        self.f_expl = vertcat(
            self.x[2],
            self.x[3],
            (
                    params.l1 ** 2
                    * params.l2
                    * params.m2
                    * self.x[2] ** 2
                    * sin(-2 * self.x[1] + 2 * self.x[0])
                    + 2 * self.u[1] * cos(-self.x[1] + self.x[0]) * params.l1
                    + 2
                    * (
                            params.g * sin(-2 * self.x[1] + self.x[0]) * params.l1 * params.m2 / 2
                            + sin(-self.x[1] + self.x[0]) * self.x[3] ** 2 * params.l1 * params.l2 * params.m2
                            + params.g * params.l1 * (params.m1 + params.m2 / 2) * sin(self.x[0])
                            - self.u[0]
                    )
                    * params.l2
            )
            / params.l1 ** 2
            / params.l2
            / (params.m2 * cos(-2 * self.x[1] + 2 * self.x[0]) - 2 * params.m1 - params.m2),
            (
                    -params.g
                    * params.l1
                    * params.l2
                    * params.m2
                    * (params.m1 + params.m2)
                    * sin(-self.x[1] + 2 * self.x[0])
                    - params.l1
                    * params.l2 ** 2
                    * params.m2 ** 2
                    * self.x[3] ** 2
                    * sin(-2 * self.x[1] + 2 * self.x[0])
                    - 2
                    * self.x[2] ** 2
                    * params.l1 ** 2
                    * params.l2
                    * params.m2
                    * (params.m1 + params.m2)
                    * sin(-self.x[1] + self.x[0])
                    + 2 * self.u[0] * cos(-self.x[1] + self.x[0]) * params.l2 * params.m2
                    + params.l1
                    * (params.m1 + params.m2)
                    * (sin(self.x[1]) * params.g * params.l2 * params.m2 - 2 * self.u[1])
            )
            / params.l2 ** 2
            / params.l1
            / params.m2
            / (params.m2 * cos(-2 * self.x[1] + 2 * self.x[0]) - 2 * params.m1 - params.m2)
        )


class TriplePendulumModel(AbstractModel):
    def __init__(self, params):
        super().__init__(params)

    def addDynamicsModel(self, params):
        """ Define the triple pendulum dynamics model. """
        self.model.name = "triple_pendulum"

        self.x = SX.sym("x", 6)
        self.x_dot = SX.sym("x_dot", 6)
        self.u = SX.sym("u", 3)
        self.p = SX.sym("p", 1)

        # dynamics
        self.f_expl = vertcat(
            self.x[3],
            self.x[4],
            self.x[5],
            (-params.g * params.l1 * params.l2 * params.l2 * params.m1 * params.m3 * sin(
                -2 * self.x[2] + 2 * self.x[1] + self.x[
                    0]) - params.g * params.l1 * params.l2 * params.l2 * params.m1 * params.m3 * sin(
                2 * self.x[2] - 2 * self.x[1] + self.x[0]) + 2 * self.u[0] * params.l2 * params.l2 * params.m3 * cos(
                -2 * self.x[2] + 2 * self.x[1]) + 2 * self.x[
                 3] ** 2 * params.l1 ** 2 * params.l2 * params.l2 * params.m2 * (
                     params.m2 + params.m3) * sin(-2 * self.x[1] + 2 * self.x[0]) - 2 * self.u[
                 2] * params.l1 * params.l2 * (
                     params.m2 + params.m3) * cos(
                -2 * self.x[1] + self.x[0] + self.x[2]) - 2 * self.u[1] * params.l1 * params.l2 * params.m3 * cos(
                -2 * self.x[2] + self.x[1] + self.x[
                    0]) + 2 * params.l1 * params.l2 * params.l2 ** 2 * params.m2 * params.m3 * self.x[5] ** 2 * sin(
                -2 * self.x[1] + self.x[0] + self.x[2]) + 2 * self.u[2] * params.l1 * params.l2 * (
                     params.m2 + params.m3) *
             cos(self.x[0] - self.x[2]) + 2 * (
                     self.u[1] * params.l1 * (params.m3 + 2 * params.m2) * cos(-self.x[1] + self.x[0]) + (
                     params.g * params.l1 * params.m2 * (params.m2 + params.m3) * sin(
                 -2 * self.x[1] + self.x[0]) + 2 * self.x[4] ** 2 * params.l1 * params.l2 * params.m2 * (
                             params.m2 + params.m3) * sin(-self.x[1] + self.x[0]) + params.m3 * self.x[
                         5] ** 2 * sin(
                 self.x[0] - self.x[2]) * params.l1 * params.l2 * params.m2 + params.g * params.l1 * (
                             params.m2 ** 2 + (
                             params.m3 + 2 * params.m1) * params.m2 + params.m1 * params.m3) * sin(
                 self.x[0]) - self.u[0] * (
                             params.m3 + 2 * params.m2)) * params.l2) * params.l2) / params.l1 ** 2 / params.l2 / (
                    params.m2 * (params.m2 + params.m3) * cos(
                -2 * self.x[1] + 2 * self.x[0]) + params.m1 * params.m3 * cos(
                -2 * self.x[2] + 2 * self.x[1]) - params.m2 ** 2 + (
                            -params.m3 - 2 * params.m1) * params.m2 - params.m1 * params.m3) / params.l2 / 2,
            (-2 * self.u[2] * params.l1 * params.l2 * (params.m2 + params.m3) * cos(
                2 * self.x[0] - self.x[2] - self.x[
                    1]) - 2 * params.l1 * params.l2 * params.l2 ** 2 * params.m2 * params.m3 * self.x[5] ** 2 * sin(
                2 * self.x[0] - self.x[2] - self.x[
                    1]) + params.g * params.l1 * params.l2 * params.l2 * params.m1 * params.m3 * sin(
                self.x[1] + 2 * self.x[0] - 2 * self.x[2]) - params.g * params.l1 * params.l2 * (
                     (params.m1 + 2 * params.m2) * params.m3 + 2 * params.m2 * (
                     params.m1 + params.m2)) * params.l2 * sin(
                -self.x[1] + 2 * self.x[0]) - 2 * self.x[
                 4] ** 2 * params.l1 * params.l2 ** 2 * params.l2 * params.m2 * (
                     params.m2 + params.m3) * sin(
                -2 * self.x[1] + 2 * self.x[0]) + 2 * self.u[1] * params.l1 * params.l2 * params.m3 * cos(
                -2 * self.x[2] + 2 * self.x[0]) + 2 * params.l1 * params.l2 ** 2 * params.l2 * params.m1 * params.m3 *
             self.x[4] ** 2 * sin(
                        -2 * self.x[2] + 2 * self.x[1]) - 2 * self.u[0] * params.l2 * params.l2 * params.m3 * cos(
                        -2 * self.x[2] + self.x[1] + self.x[
                            0]) + 2 * params.l1 ** 2 * params.l2 * params.l2 * params.m1 * params.m3 * self.x[
                 3] ** 2 * sin(
                        -2 * self.x[2] +
                        self.x[1] + self.x[0]) - 2 * params.l1 ** 2 * params.l2 * self.x[3] ** 2 * (
                     (params.m1 + 2 * params.m2) * params.m3 + 2 * params.m2 * (
                     params.m1 + params.m2)) * params.l2 * sin(
                        -self.x[1] + self.x[0]) + 2 * self.u[2] * params.l1 * params.l2 * (
                     params.m3 + 2 * params.m1 + params.m2) * cos(
                        -self.x[2] + self.x[1]) + (2 * self.u[0] * params.l2 * (params.m3 + 2 * params.m2) * cos(
                        -self.x[1] + self.x[0]) + params.l1 * (
                                                           4 * self.x[5] ** 2 * params.m3 * params.l2 * (
                                                           params.m1 + params.m2 / 2) * params.l2 * sin(
                                                       -self.x[2] + self.x[
                                                           1]) + params.g * params.m3 * params.l2 * params.m1 * sin(
                                                       -2 * self.x[2] + self.x[1]) + params.g * (
                                                                   (
                                                                           params.m1 + 2 * params.m2) * params.m3 + 2 * params.m2 * (
                                                                           params.m1 + params.m2)) * params.l2 * sin(
                                                       self.x[1]) - 2 * self.u[1] * (
                                                                   params.m3 + 2 * params.m1 + 2 * params.m2))) * params.l2) / (
                    params.m2 * (params.m2 + params.m3) * cos(
                -2 * self.x[1] + 2 * self.x[0]) + params.m1 * params.m3 * cos(
                -2 * self.x[2] + 2 * self.x[1]) + (
                            -params.m1 - params.m2) * params.m3 - 2 * params.m1 * params.m2 - params.m2 ** 2) / params.l1 / params.l2 / params.l2 ** 2 / 2,
            (-2 * params.m3 * self.u[1] * params.l1 * params.l2 * (params.m2 + params.m3) * cos(
                2 * self.x[0] - self.x[2] - self.x[
                    1]) + params.g * params.m3 * params.l1 * params.l2 * params.l2 * params.m1 * (
                     params.m2 + params.m3) * sin(2 * self.x[0] + self.x[2] - 2 * self.x[1]) + 2 * self.u[
                 2] * params.l1 * params.l2 * (
                     params.m2 + params.m3) ** 2 * cos(
                -2 * self.x[1] + 2 * self.x[
                    0]) - params.g * params.m3 * params.l1 * params.l2 * params.l2 * params.m1 * (
                     params.m2 + params.m3) * sin(
                2 * self.x[0] - self.x[2]) - params.g * params.m3 * params.l1 * params.l2 * params.l2 * params.m1 * (
                     params.m2 + params.m3) * sin(
                -self.x[2] + 2 * self.x[1]) - 2 * params.l1 * params.l2 * params.l2 ** 2 * params.m1 * params.m3 ** 2 *
             self.x[5] ** 2 * sin(
                        -2 * self.x[2] + 2 * self.x[1]) - 2 * self.u[0] * params.l2 * params.l2 * params.m3 * (
                     params.m2 + params.m3) * cos(
                        -2 * self.x[1] + self.x[0] + self.x[2]) + 2 * params.m3 * self.x[3] ** 2 * params.l1 ** 2 *
             params.l2 * params.l2 * params.m1 * (params.m2 + params.m3) * sin(
                        -2 * self.x[1] + self.x[0] + self.x[2]) + 2 * params.m3 * self.u[1] * params.l1 * params.l2 * (
                     params.m3 + 2 * params.m1 + params.m2) * cos(-self.x[2] + self.x[1]) + (params.m2 + params.m3) * (
                     2 * self.u[0] * params.l2 * params.m3 * cos(self.x[0] - self.x[2]) + params.l1 * (
                     -2 * params.m3 * self.x[3] ** 2 * params.l1 * params.l2 * params.m1 * sin(
                 self.x[0] - self.x[2]) - 4 * params.m3 * self.x[4] ** 2 * sin(
                 -self.x[2] + self.x[1]) * params.l2 * params.l2 * params.m1 + params.g * params.m3 * sin(
                 self.x[2]) * params.l2 * params.m1 - 2 * self.u[2] * (
                             params.m3 + 2 * params.m1 + params.m2))) * params.l2) / params.m3 / (
                    params.m2 * (params.m2 + params.m3) * cos(
                -2 * self.x[1] + 2 * self.x[0]) + params.m1 * params.m3 * cos(
                -2 * self.x[2] + 2 * self.x[1]) + (
                            -params.m1 - params.m2) * params.m3 - 2 * params.m1 * params.m2 - params.m2 ** 2) / params.l1 / params.l2 ** 2 / params.l2 / 2,
        )
