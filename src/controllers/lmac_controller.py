from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.meta import REGISTRY as meta_REGISTRY
import torch as th

import time

class LMAC_MAC:
    def __init__(self, scheme, groups, comm_modules, message_dims, important_state, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)

        self.comms =[m.communication for m in comm_modules]
        self.message_dims = message_dims

        self.important_state_dim = len(important_state)

        self._build_agents(input_shape + self.args.latent_dim, self.important_state_dim)
        
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

        self.msg_hidden_states = None
        self.without_msg_hidden_states = None

        self.imp_state = important_state

        self.meta = meta_REGISTRY[args.meta](input_shape + sum(self.message_dims), self.important_state_dim, args)
    
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        agent_outputs, *_  = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_inputs_to_No_msg = self._build_inputs(ep_batch, t)
        m_inputs = self._build_time_inputs(ep_batch, t)
        if agent_inputs.shape[0]/ep_batch.batch_size >=1:
            agent_inputs = agent_inputs.reshape(ep_batch.batch_size, self.n_agents, -1)
        else: 
            agent_inputs = agent_inputs.unsqueeze(0)

        if m_inputs.shape[0]/ep_batch.batch_size >=1:
            m_inputs = m_inputs.reshape(ep_batch.batch_size,self.args.time_seq ,self.n_agents, -1)
        else:
            m_inputs = m_inputs.unsqueeze(0)

        phase_messages = []
        for i, comm in enumerate(self.comms):
            if i < 2:
                phase_messages.append(comm(agent_inputs).to(agent_inputs.device))
            else:
                phase_messages.append(comm(m_inputs).to(agent_inputs.device))

        for i, msg in enumerate(phase_messages):
            if agent_inputs.shape[0] / ep_batch.batch_size >= 1:
                phase_messages[i] = msg.reshape(ep_batch.batch_size * self.n_agents, -1)
            else:
                phase_messages[i] = msg.squeeze(0)
            if i >= 1:
                phase_messages[i] = phase_messages[i][:, agent_inputs.shape[-1]:].to(agent_inputs.device)

        messages = th.cat(phase_messages, dim=-1)

        pred_state_meta_combined, latent_z, self.msg_hidden_states, latent_recon = self.meta(messages, self.msg_hidden_states)
        pred_state, pred_meta = pred_state_meta_combined.split(self.important_state_dim, dim=-1)
        
        pred_state_to_agent = pred_state.detach() 
        latent_to_agent = latent_z.detach()  
        agent_inputs = th.cat([agent_inputs_to_No_msg, latent_to_agent], dim=-1)
        
        agent_outs, self.hidden_states = self.agent(agent_inputs,self.hidden_states)
        avail_actions = ep_batch["avail_actions"][:, t]

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        
        return (
            agent_outs.view(ep_batch.batch_size, self.n_agents, -1),
            pred_state.view(ep_batch.batch_size, self.n_agents, -1),
            pred_meta.view(ep_batch.batch_size, self.n_agents, -1),
            latent_z.view(ep_batch.batch_size, self.n_agents, -1),
            latent_recon.view(ep_batch.batch_size, self.n_agents, -1),
        )
    
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.msg_hidden_states = self.meta.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
    
    def parameters(self):
        params = list(self.agent.parameters()) + list(self.meta.parameters())
        return params

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.meta.load_state_dict(other_mac.meta.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.meta.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.meta.state_dict(), "{}/meta.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.meta.load_state_dict(th.load("{}/meta.th".format(path), map_location=lambda storage, loc: storage))
        
    def _build_agents(self, input_shape, important_state_dim):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, important_state_dim ,self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _build_state_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["state"][:, t].unsqueeze(1).repeat(1, self.n_agents, 1))  # b1av
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
    
    def _build_time_inputs(self, batch, t):
        bs = batch.batch_size
        time_seq = self.args.time_seq 
        input_seq = []
        for i in range(t - time_seq + 1, t + 1):
            # observation
            if i < 0:
                obs = th.full_like(batch["obs"][:, 0], fill_value=-1.0)  
            else:
                obs = batch["obs"][:, i] 
            input_seq.append(obs)

            # last action
            if self.args.obs_last_action:
                if i <= 0:
                    last_action = th.full_like(batch["actions_onehot"][:, 0], fill_value=-1.0)
                else:
                    last_action = batch["actions_onehot"][:, i-1]
                input_seq.append(last_action)

            if self.args.obs_agent_id:
                agent_id = th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
                input_seq.append(agent_id)

        input_seq = [x.reshape(bs * self.n_agents, -1) for x in input_seq]
        inputs = th.cat(input_seq, dim=1)  
        return inputs