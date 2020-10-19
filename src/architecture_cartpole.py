from architecture import * 


class Trainer(Trainer):
    """ The class contains the training logic """

    def __init__(self, world_model, robustness_computer, \
                attacker_nn, defender_nn, logging_dir=None):
        super().__init__(world_model, robustness_computer, attacker_nn, defender_nn)

    def train_attacker_step(self, time_horizon, dt, atk_static):
        """ Training step for the attacker. The defender's passive. """
        z = torch.rand(self.attacker.noise_size).float()
        oe = torch.tensor(self.model.environment.status).float()
        oa = torch.tensor(self.model.agent.status).float()

        atk_policy = self.attacker(torch.cat((z, oe)))

        with torch.no_grad():
            def_policy = self.defender(oa)

        t = 0
        for i in range(time_horizon):
            
            theta = self.model.environment.status[1]
            if torch.abs(theta)<1.:

                # if the attacker is static (e.g. in the case it does not vary over time)
                # the policy function is always sampled in the same point since the
                # attacker do not vary policy over time
                atk_input = atk_policy(0 if atk_static else t)
                def_input = def_policy(t)

                self.model.step(atk_input, def_input, dt)

                t += dt
                loss_penalty = 0 

            else:
                loss_penalty = +10

        rho = self.robustness_computer.compute(self.model)

        self.attacker_optimizer.zero_grad()
        loss = self.attacker_loss_fn(rho) + loss_penalty
        loss.backward()

        self.attacker_optimizer.step()
        return loss.detach().float()

    def train_defender_step(self, time_horizon, dt, atk_static):
        """ Training step for the defender. The attacker's passive. """
        z = torch.rand(self.attacker.noise_size).float()
        oa = torch.tensor(self.model.agent.status).float()
        oe = torch.tensor(self.model.environment.status).float()

        with torch.no_grad():
            atk_policy = self.attacker(torch.cat((z, oe)))

        def_policy = self.defender(oa)

        t = 0
        for i in range(time_horizon):

            theta = self.model.agent.status[1]
            if torch.abs(theta)<1.:

                # if the attacker is static, see the comments above
                atk_input = atk_policy(0 if atk_static else t)
                def_input = def_policy(t)

                self.model.step(atk_input, def_input, dt)
                t += dt
                loss_penalty = 0 

            else:
                loss_penalty = +10

        rho = self.robustness_computer.compute(self.model)

        self.defender_optimizer.zero_grad()
        loss = self.defender_loss_fn(rho) + loss_penalty
        loss.backward()

        self.defender_optimizer.step()
        return loss.detach().float()