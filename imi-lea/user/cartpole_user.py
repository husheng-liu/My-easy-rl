import math

import numpy as np

from gym.envs.classic_control import CartPoleEnv


class CartPoleUserEnv(CartPoleEnv):
	def __init__(self):
		super().__init__()
		self.steps_beyond_terminated = None

	def expert_step(self, action):
		err_msg = f"{action!r} ({type(action)}) invalid"
		assert self.action_space.contains(action), err_msg
		assert self.state is not None, "Call reset before using step method."
		x, x_dot, theta, theta_dot = self.state
		force = self.force_mag if action == 1 else -self.force_mag
		costheta = math.cos(theta)
		sintheta = math.sin(theta)

		# For the interested reader:
		# https://coneural.org/florian/papers/05_cart_pole.pdf
		temp = (
				       force + self.polemass_length * theta_dot ** 2 * sintheta
		       ) / self.total_mass
		thetaacc = (self.gravity * sintheta - costheta * temp) / (
				self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
		)
		xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

		if self.kinematics_integrator == "euler":
			x = x + self.tau * x_dot
			x_dot = x_dot + self.tau * xacc
			theta = theta + self.tau * theta_dot
			theta_dot = theta_dot + self.tau * thetaacc
		else:  # semi-implicit euler
			x_dot = x_dot + self.tau * xacc
			x = x + self.tau * x_dot
			theta_dot = theta_dot + self.tau * thetaacc
			theta = theta + self.tau * theta_dot

		terminated = bool(
			x < -self.x_threshold
			or x > self.x_threshold
			or theta < -self.theta_threshold_radians
			or theta > self.theta_threshold_radians
		)

		if not terminated:
			reward = 1.0
		elif self.steps_beyond_terminated is None:
			# Pole just fell!
			self.steps_beyond_terminated = 0
			reward = 1.0
		else:
			self.steps_beyond_terminated += 1
			reward = 0.0

		return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
