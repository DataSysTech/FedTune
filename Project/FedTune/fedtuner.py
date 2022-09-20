import copy


class FedTuner:

    def __init__(self, *, alpha: float, beta: float, gamma: float, delta: float, initial_M: int, initial_E: float,
                 M_min: int, M_max: int, E_min: float, E_max: float, penalty: float):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.initial_M = initial_M
        self.initial_E = initial_E
        self.M_min = M_min
        self.M_max = M_max
        self.E_min = E_min
        self.E_max = E_max
        self.penalty = penalty

        self.initial_number = 10 ** 9
        self.eta_t = self.initial_number
        self.eta_q = self.initial_number
        self.eta_z = self.initial_number
        self.eta_v = self.initial_number
        self.zeta_t = self.initial_number
        self.zeta_q = self.initial_number
        self.zeta_z = self.initial_number
        self.zeta_v = self.initial_number

        # skip the first training rounds
        self.n_round_skipped = 1

        # FL settings
        self.S_cur = FLSetting()  # Current FL hyper-parameter set
        self.S_cur.M = self.initial_M
        self.S_cur.E = self.initial_E

        self.S_prv = FLSetting()        # Previous FL hyper-parameter set
        self.S_prv_save = FLSetting()   # Save Previous FL for updating eta and zeta

        # decision only made when accuracy is improved by at least eps_accuracy
        self.eps_accuracy = 0.01

    def update(self, *, model_accuracy: float, compT: float, transT: float, compL: float, transL: float) -> tuple[int, float]:

        self.S_cur.add_cost(compT=compT, transT=transT, compL=compL, transL=transL)

        next_M = self.S_cur.M
        next_E = self.S_cur.E

        if model_accuracy - self.S_prv.model_accuracy > self.eps_accuracy:

            self.S_cur.normalize_cost(accuracy_length=model_accuracy-self.S_prv.model_accuracy)

            if self.n_round_skipped <= 0:

                # Calculate the improvement ratio
                g = self.alpha * (self.S_cur.compT - self.S_prv.compT) / self.S_cur.compT \
                    + self.beta * (self.S_cur.transT - self.S_prv.transT) / self.S_cur.transT \
                    + self.gamma * (self.S_cur.compL - self.S_prv.compL) / self.S_cur.compL \
                    + self.delta * (self.S_cur.transT - self.S_cur.transL) / self.S_cur.transL

                # only update when they move towards preference
                if self.S_cur.M > self.S_prv.M:
                    self.eta_t = abs(self.S_cur.compT - self.S_prv.compT) / abs(self.S_prv.compT - self.S_prv_save.compT)
                    self.eta_q = abs(self.S_cur.transT - self.S_prv.transT) / abs(self.S_prv.transT - self.S_prv_save.transT)
                    if g > 0:
                        self.eta_z *= self.penalty
                        self.eta_v *= self.penalty
                else:
                    self.eta_z = abs(self.S_cur.compL - self.S_prv.compL) / abs(self.S_prv.compL - self.S_prv_save.compL)
                    self.eta_v = abs(self.S_cur.transL - self.S_prv.transL) / abs(self.S_prv.transL - self.S_prv_save.transL)
                    if g > 0:
                        self.eta_t *= self.penalty
                        self.eta_q *= self.penalty
                if self.S_cur.E > self.S_prv.E:
                    self.zeta_q = abs(self.S_cur.transT - self.S_prv.transT) / abs(self.S_prv.transT - self.S_prv_save.transT)
                    self.zeta_v = abs(self.S_cur.transL - self.S_prv.transL) / abs(self.S_prv.transL - self.S_prv_save.transL)
                    if g > 0:
                        self.zeta_t *= self.penalty
                        self.zeta_z *= self.penalty
                else:
                    self.zeta_t = abs(self.S_cur.compT - self.S_prv.compT) / abs(self.S_prv.compT - self.S_prv_save.compT)
                    self.zeta_z = abs(self.S_cur.compL - self.S_prv.compL) / abs(self.S_prv.compL - self.S_prv_save.compL)
                    if g > 0:
                        self.zeta_q *= self.penalty
                        self.zeta_v *= self.penalty

                delta_M = self.alpha * self.eta_t * abs(self.S_cur.compT - self.S_prv.compT) / self.S_cur.compT \
                          + self.beta * self.eta_q * abs(self.S_cur.transT - self.S_prv.transT) / self.S_cur.transT \
                          - self.gamma * self.eta_z * abs(self.S_cur.compL - self.S_prv.compL) / self.S_cur.compL \
                          - self.delta * self.eta_v * abs(self.S_cur.transL - self.S_prv.transL) / self.S_cur.transL

                delta_E = -self.alpha * self.zeta_t * abs(self.S_cur.compT - self.S_prv.compT) / self.S_cur.compT \
                          + self.beta * self.zeta_q * abs(self.S_cur.transT - self.S_prv.transT) / self.S_cur.transT \
                          - self.gamma * self.zeta_z * abs(self.S_cur.compL - self.S_prv.compL) / self.S_cur.compL \
                          + self.delta * self.zeta_v * abs(self.S_cur.transL - self.S_prv.transL) / self.S_cur.transL

                if delta_M > 0:
                    next_M = self.S_cur.M + 1
                    next_M = min(next_M, self.M_max)
                else:
                    next_M = self.S_cur.M - 1
                    next_M = max(next_M, self.M_min)

                if delta_E > 0:
                    next_E = self.S_cur.E + 1
                    next_E = min(float(next_E), self.E_max)
                else:
                    next_E = self.S_cur.E - 1
                    next_E = max(float(next_E), self.E_min)
            else:
                # skip the first training rounds
                self.n_round_skipped -= 1

            self.S_cur.model_accuracy = model_accuracy

            self.S_prv_save = copy.deepcopy(self.S_prv)
            self.S_prv = copy.deepcopy(self.S_cur)
            self.S_cur = FLSetting()
            self.S_cur.M = next_M
            self.S_cur.E = next_E

        return next_M, next_E

    def get_eta_and_zeta(self) -> tuple[float, float, float, float, float, float, float, float]:
        """ return hyper-parameters for gains

        :return: hyper-parameters for gains
        """

        return self.eta_t, self.eta_q, self.eta_z, self.eta_v, self.zeta_t, self.zeta_q, self.zeta_z, self.zeta_v


class FLSetting:
    # Store information under a set of FL hyper-parameters
    def __init__(self):
        self.compT = 0
        self.transT = 0
        self.compL = 0
        self.transL = 0
        self.model_accuracy = 0

        self.M = -1
        self.E = -1

    def add_cost(self, *, compT: float, transT: float, compL: float, transL: float) -> None:
        self.compT += compT
        self.transT += transT
        self.compL += compL
        self.transL += transL

    def normalize_cost(self, *, accuracy_length) -> None:
        self.compT /= accuracy_length
        self.transT /= accuracy_length
        self.compL /= accuracy_length
        self.transL /= accuracy_length

    def __str__(self):
        return f'self.compT = {self.compT}, ' \
               f'self.transT = {self.transT}, ' \
               f'self.compL = {self.compL}, ' \
               f'self.transL = {self.transL}'
