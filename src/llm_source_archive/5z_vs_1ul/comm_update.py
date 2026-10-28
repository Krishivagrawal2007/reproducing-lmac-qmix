import torch as th

def message_design_instruction():
    """
    Improved Message Design Instruction for 5z_vs_1ul (SMAC):

    1. **Purpose**: 
       This protocol enables full-team coordination for kiting, focus fire, and survivability by explicitly sharing each agent's absolute position, shield, and intent cues (ready flag), as well as opportunistically sharing enemy state if observed.
    
    2. **Message Content & Justification**:
       **Each agent broadcasts a message containing:**

       [A] **Own Absolute Position** (2 floats):  
           - Absolute X, Y coordinates derived from observed relative positions of all visible allies and self.  
           - *Rationale*: Allows agents to reconstruct the team's formation, support kiting, and synchronize movement even under partial observability.

       [B] **Own Shield Ratio** (1 float):  
           - *Rationale*: Enables the team to prioritize protection and identify vulnerable agents (Health is in `comm_init`).

       [C] **Own "Ready to Engage" Signal** (1 float, binary):  
           - 1 if agent is in position to attack (can see Ultralisk and is not moving), else 0.  
           - *Rationale*: Facilitates precise timing for coordinated attacks or retreats.

       [D] **Ultralisk Observation** (5 floats):  
           - If Ultralisk visible: absolute X, Y (calculated via own absolute position + relative), health ratio, shield ratio, and 'enemy_seen' (1).  
           - Else: zeros.  
           - *Rationale*: Enables global enemy state reconstruction, supporting focus fire and joint kiting.

       **Total message_dim = 2 (own XY) + 1 (shield) + 1 (ready) + 5 (Ultralisk obs) = 9**

    3. **Communication Protocol**:
       - **Broadcast**: Each agent sends its message to all others (not to itself).
       - Each agent receives messages from the other 4 agents (not itself), resulting in 4 × 9 = 36 message dimensions appended to its own observation.
       - Messages are concatenated in sender order for each receiver.

    4. **Explicitness & Compactness**:
       - Each field has a fixed semantic order.
       - Only shares information that cannot be reliably inferred from local observations.
       - Excludes redundant fields present in previous design.

    5. **Why these fields?**
       - **Absolute positions** are necessary for global team reasoning.
       - **Status (Shield) and intent (Ready)** support dynamic coordination.
       - **Ultralisk absolute state** is essential for focus fire.

    6. **Efficiency**:
       - All operations are batch- and vectorized; no explicit for-loops over batch or agent dimension.
       - No trainable parameters.

    7. **Resulting Output**:
       - Output tensor shape: (batch, 5, 48 + 36).

    """
    return (
        "Each agent broadcasts a message composed of: "
        "[Own absolute X, Own absolute Y, Own shield ratio, "
        "Ready-to-engage flag (1 if can attack Ultralisk and not moving, else 0), "
        "Ultralisk absolute X, Ultralisk absolute Y, Ultralisk health ratio, Ultralisk shield ratio, Ultralisk seen (1/0)]. "
        "Each agent receives the 4 messages from teammates (not including itself), concatenating them to its own observation. "
        "Total added message_dim = 9*4 = 36. "
        "Redundant info (Health, Last Action, Sender ID) from comm_init is omitted."
    )

def communication(o):
    """
    Input:
        o: torch.Tensor, shape (batch, 5, 48), original agent observations.

    Output:
        messages_o: torch.Tensor, shape (batch, 5, 48+36), concatenated observation and received messages.
    """
    device = o.device
    dtype = o.dtype
    B, N, D = o.shape
    assert N == 5 and D == 48

    # -- 1. Compute own absolute position (X, Y) for each agent --
    # For each agent, we reconstruct own absolute position by referencing a visible ally (preferably the lowest-index visible ally).
    # If no ally is visible, use (0,0).
    # For agent i, fields for ally j (j != i) are:
    #   - Is ally visible: 10 + 6*(j-1) if j < i, 10 + 6*j if j > i
    #   - Rel X to ally:   12 + 6*(j-1) if j < i, 12 + 6*j if j > i
    #   - Rel Y to ally:   13 + 6*(j-1) if j < i, 13 + 6*j if j > i

    own_abs_xy = th.zeros((B, N, 2), device=device, dtype=dtype)
    for ref in range(N):
        # skip own index
        visible_mask = []
        rel_x = []
        rel_y = []
        for i in range(N):
            if i == ref:
                # skip (cannot see self)
                visible_mask.append(th.zeros(B, 1, device=device, dtype=dtype))
                rel_x.append(th.zeros(B, 1, device=device, dtype=dtype))
                rel_y.append(th.zeros(B, 1, device=device, dtype=dtype))
            else:
                if ref < i:
                    start = 10 + 6*(i-1)
                else:
                    start = 10 + 6*i
                visible_mask.append(o[:, ref, start:start+1])
                rel_x.append(o[:, ref, start+2:start+3])
                rel_y.append(o[:, ref, start+3:start+4])
        # For agent 'ref', stack visibility and relative positions to all other agents
        visible_mask_stack = th.cat(visible_mask, dim=-1)  # (B, N)
        rel_x_stack = th.cat(rel_x, dim=-1)                # (B, N)
        rel_y_stack = th.cat(rel_y, dim=-1)                # (B, N)

        # For agent 'ref', find first visible ally (excluding self)
        # Mask out self
        visible_mask_stack[:, ref] = 0
        # Find first visible agent (lowest index)
        has_visible = visible_mask_stack.sum(dim=-1, keepdim=True) > 0  # (B, 1)
        first_visible_idx = visible_mask_stack.float().argmax(dim=-1)   # (B,)
        # Gather rel_x/y for first visible
        rel_x_first = rel_x_stack[th.arange(B), first_visible_idx]      # (B,)
        rel_y_first = rel_y_stack[th.arange(B), first_visible_idx]      # (B,)
        # For batch elements with no visible allies, use (0,0)
        rel_x_first = rel_x_first * has_visible.squeeze(-1)
        rel_y_first = rel_y_first * has_visible.squeeze(-1)
        # For own absolute position, assume the visible ally is at (0,0), so own abs pos = -rel
        own_abs_xy[:, ref, 0] = -rel_x_first
        own_abs_xy[:, ref, 1] = -rel_y_first
        # If no visible ally, remain (0,0)
    # own_abs_xy: (B, N, 2)

    # -- 2. Own shield ratio (Health is in comm_init) --
    own_shield = o[..., 35:36]  # (B, N, 1)

    # -- 3. Ready-to-engage flag --
    # 1 if agent can see Ultralisk and is not moving (i.e., last action is Attack or Stop)
    ultra_visible = o[..., 4:5]  # (B, N, 1)
    last_attack = o[..., 42:43]  # Last Action-Attack (B, N, 1)
    last_stop = o[..., 37:38]    # Last Action-Stop
    ready_flag = ultra_visible * ((last_attack + last_stop) > 0).float()  # (B, N, 1)

    # -- 4. Ultralisk absolute X, Y, health, shield, seen --
    # If Ultralisk is visible, Ultralisk abs pos = own abs pos + rel (from dims 5,6)
    rel_ultra_x = o[..., 5:6]  # (B, N, 1)
    rel_ultra_y = o[..., 6:7]
    ultra_health = o[..., 8:9]
    ultra_shield = o[..., 9:10]
    ultra_seen_flag = ultra_visible  # (B, N, 1)
    ultra_abs_x = own_abs_xy[..., 0:1] + rel_ultra_x * ultra_visible
    ultra_abs_y = own_abs_xy[..., 1:2] + rel_ultra_y * ultra_visible
    # Zero out all if not visible
    ultra_abs_x = ultra_abs_x * ultra_visible
    ultra_abs_y = ultra_abs_y * ultra_visible
    ultra_health = ultra_health * ultra_visible
    ultra_shield = ultra_shield * ultra_visible
    ultra_seen_flag = ultra_seen_flag

    ultra_fields = th.cat([ultra_abs_x, ultra_abs_y, ultra_health, ultra_shield, ultra_seen_flag], dim=-1)  # (B, N, 5)

    # -- 5. Stack message fields (Removed redundant Health, LastAction, SenderID) --
    msg_fields = [
        own_abs_xy,           # (B, N, 2)
        own_shield,           # (B, N, 1)
        ready_flag,           # (B, N, 1)
        ultra_fields,         # (B, N, 5)
    ]
    msg = th.cat(msg_fields, dim=-1)  # (B, N, 9)

    # -- 8. For each agent, collect messages from the other 4 agents (not self) --
    mask = (1 - th.eye(N, device=device, dtype=dtype)).unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
    msg_exp = msg.unsqueeze(1).expand(B, N, N, -1)  # (B, N, N, 22)
    msg_masked = msg_exp * mask  # self-message is zeroed

    # For each receiver (agent axis=1), extract the messages from other 4 agents (axis=2)
    indices = []
    for i in range(N):
        indices.append([j for j in range(N) if j != i])
    indices = th.tensor(indices, device=device)  # (N, 4)
    indices_expand = indices.unsqueeze(0).expand(B, -1, -1)  # (B, N, 4)
    indices_expand = indices_expand.unsqueeze(-1).expand(-1, -1, -1, 9)  # (B, N, 4, 9)
    msgs_from_others = th.gather(msg_masked, 2, indices_expand)  # (B, N, 4, 9)

    # Reshape to (batch, 5, 36)
    msgs_from_others = msgs_from_others.reshape(B, N, 4*9)  # (B, N, 36)

    # -- 9. Concatenate with original observation --
    messages_o = th.cat([o, msgs_from_others], dim=-1)  # (B, N, 48+36)

    return messages_o
