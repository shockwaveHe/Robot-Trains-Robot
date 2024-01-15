class BaseController:
    def compute_control(self, desired_state, current_state):
        raise NotImplementedError

    def update_parameters(self, params):
        raise NotImplementedError
