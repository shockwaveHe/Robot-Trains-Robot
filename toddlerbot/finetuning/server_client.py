import pickle
import socket
import threading
import json
import time
import numpy as np
import io
import base64
import torch

def state_dict_to_base64(state_dict):
    """
    Serializes the state dict to a base64–encoded string.
    """
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    b64_str = base64.b64encode(buffer.read()).decode('utf-8')
    return b64_str

def base64_to_state_dict(b64_str, device='cpu'):
    """
    Converts a base64–encoded string back into a state dict.
    The state dict is loaded onto the specified device.
    """
    buffer = io.BytesIO(base64.b64decode(b64_str))
    state_dict = torch.load(buffer, map_location=device)
    return state_dict

def dump_experience_to_base64(exp_data: dict) -> str:
    """
    Serializes the experience data (a dict) into a base64–encoded string.
    """
    buffer = io.BytesIO()
    pickle.dump(exp_data, buffer)
    buffer.seek(0)
    b64_str = base64.b64encode(buffer.read()).decode('utf-8')
    return b64_str

def load_experience_from_base64(b64_str: str) -> dict:
    """
    Converts a base64–encoded string back into the original experience data dict.
    """
    buffer = io.BytesIO(base64.b64decode(b64_str))
    exp_data = pickle.load(buffer)
    return exp_data

class RemoteClient:
    def __init__(self, server_ip, server_port, exp_folder):
        self.server_addr = (server_ip, server_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.server_addr)
        try:
            self.sock.sendall(f"{exp_folder}\n".encode('utf-8'))
        except Exception as e:
            print("Error sending hello message:", e)
        self.new_state_dict = None
        self.ready_to_update = False
        print(f"Connected to learner at {server_ip}:{server_port}")
        threading.Thread(target=self._receive_policy_updates, daemon=True).start()
    
    def send_experience(self, data: dict):
        msg = json.dumps(data) + "\n"
        try:
            self.sock.sendall(msg.encode('utf-8'))
        except Exception as e:
            print("Error sending experience:", e)
    
    def _receive_policy_updates(self):
        buffer = ""
        while True:
            try:
                data = self.sock.recv(4096)
                if not data:
                    print("Connection closed by learner")
                    break
                buffer += data.decode('utf-8')
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line:
                        msg = json.loads(line)
                        if msg.get('type') == 'policy_update':
                            # Here we call the callback with the new parameters.
                            received_b64 = msg['params_b64']
                            self.new_state_dict = base64_to_state_dict(received_b64)
                            self.ready_to_update = True
            except Exception as e:
                print("Error receiving policy update:", e)
                break


class RemoteServer:
    def __init__(self, host, port, policy):
        self.host = host
        self.port = port
        self.exp_folder = None
        self.policy = policy
        self.clients = []   # list of (conn, addr)
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((host, port))
        self.server.listen(5)
        self.is_running = True
        print(f"RemoteBufferServer listening on {host}:{port}")

    def handle_client(self, conn, addr, replay_buffer):
        print("Connected by", addr)
        # Define dimensions based on your optimizations.
        obs_frame_dim = 83    # latest frame dimension for obs
        priv_frame_dim = 122  # latest frame dimension for privileged obs
        num_frames = 15       # number of frames to stack

        # Initialize the stacks with zeros.
        obs_history = np.zeros(num_frames * obs_frame_dim, dtype=np.float32)
        privileged_obs_history = np.zeros(num_frames * priv_frame_dim, dtype=np.float32)
        buffer = ""

        while True:
            try:
                data = conn.recv(65536)
                if not data:
                    print(f"Client {addr} disconnected")
                    self.is_running = False
                    break
                buffer += data.decode('utf-8')
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line:
                        try:
                            msg = json.loads(line)
                        except Exception as e:
                            print("JSON decode error:", e)
                            continue
                        if msg.get('type') == 'experience':
                            msg = load_experience_from_base64(msg['data_b64'])
                            # Extract the latest frame for obs and privileged obs.
                            latest_obs = msg['s']   # shape: (83,)
                            latest_priv = msg['s_p']   # shape: (126,)
                            
                            # If a done/truncated flag is True, reset the stacks.
                            if msg.get('done') or msg.get('truncated'):
                                obs_history = np.zeros(num_frames * obs_frame_dim, dtype=np.float32)
                                privileged_obs_history = np.zeros(num_frames * priv_frame_dim, dtype=np.float32)
                                self.policy.last_action = np.zeros(self.policy.num_action, dtype=np.float32)
                                self.policy.last_last_action = np.zeros(self.policy.num_action, dtype=np.float32)
                            else:
                                # Shift stacks left and append the new frame at the end.

                                obs_history = np.roll(obs_history, latest_obs.size)
                                obs_history[: latest_obs.size] = latest_obs
                                privileged_obs_history = np.roll(
                                    privileged_obs_history, latest_priv.size
                                )
                                privileged_obs_history[: latest_priv.size] = latest_priv

                            # Use the original OnlineReplayBuffer's store method.
                            raw_obs = msg['raw_obs']
                            self.policy.sim.set_motor_angles(raw_obs.motor_pos)
                            self.policy.sim.forward()
                            feet_pos = self.policy.sim.get_feet_pos()
                            feet_y_dist = feet_pos["left"][1] - feet_pos["right"][1]
                            raw_obs.feet_y_dist = feet_y_dist
                            reward_dict = self.policy._compute_reward(raw_obs, msg['a'])
                            reward = sum(reward_dict.values()) * self.policy.control_dt
                            print(f"Calculated reward: {reward}, Feet y dist: {feet_y_dist}")
                            self.policy.last_last_action = self.policy.last_action.copy()
                            self.policy.last_action = msg['a'].copy()
                            self.policy.logger.log_step(
                                reward_dict,
                                raw_obs,
                                reward=reward,
                                feet_dist=feet_y_dist,
                                # walk_command=control_inputs["walk_x"],
                            )
                            replay_buffer.store(
                                s=obs_history,
                                s_p=privileged_obs_history,
                                a=msg['a'],
                                r=reward,
                                done=msg['done'],
                                truncated=msg['truncated'],
                                raw_obs=raw_obs
                            )
            except Exception as e:
                print(f"Error handling client {addr}: {e}")
                import traceback
                traceback.print_exc()
                self.is_running = False
                break
        
        conn.close()
        self.server.close()
        print("Connection closed by", addr)

    def start_receiving_data(self):
        """
        Start accepting connections and receiving experience data.
        The received data is processed and the full (stacked) state is stored in replay_buffer.
        """
        try:
            conn, addr = self.server.accept()
            data = conn.recv(1024)
            if data:
                self.exp_folder = data.decode('utf-8').strip()
                print(f"Accepted connection from {addr} for experiment {self.exp_folder}")
            else:
                print("Received empty message from", addr)
            self.clients.append((conn, addr))
            client_thread = threading.Thread(
                target=self.handle_client, args=(conn, addr, self.policy.replay_buffer), daemon=True
            )
            client_thread.start()
        except Exception as e:
            print(f"Error accepting connection: {e}")
            self.is_running = False

    def push_policy_parameters(self, policy_state_dict):
        """
        Push the latest policy parameters to all connected agents.
        The policy_state_dict must be JSON–serializable (or converted appropriately).
        """
        b64_state = state_dict_to_base64(policy_state_dict)
        payload = {
            'type': 'policy_update',
            'params_b64': b64_state
        }
        msg = json.dumps(payload) + "\n"
        for conn, addr in self.clients:
            try:
                start_time = time.time()
                conn.sendall(msg.encode('utf-8'))
                end_time = time.time()
                print(f"Sent policy update to {addr} in {(end_time - start_time) * 1000:.2f} ms")
            except Exception as e:
                print(f"Error sending policy update to {addr}: {e}")
