import numpy as np
from swarm_env import RationalSwarmForagingEnv

def test_env():
    env = RationalSwarmForagingEnv(n_agents=3, grid_size=5, num_food=2)
    obs, info = env.reset()
    print("Initial Positions:", env.agent_positions)
    
    # Force a collision scenario
    # Agent 0 moves Right, Agent 1 moves Left, into the same square
    # Agent 2 passes
    
    # Base is at 2,2. Everyone starts at 2,2. 
    # Let's move them out first.
    env.step({"agent_0": 0, "agent_1": 1, "agent_2": 2}) # Up, Down, Left
    print("Positions after move 1:", env.agent_positions)
    print("Modes:", env.agent_modes)
    
    # Now try to collide agent 0 and 1 into center?
    # agent 0 is at 2,1. Moves Down (1)
    # agent 1 is at 2,3. Moves Up (0)
    # agent 2 is at 1,2. Passes (4)
    obs, rewards, term, trunc, info = env.step({"agent_0": 1, "agent_1": 0, "agent_2": 4})
    print("Positions after collision attempt:", env.agent_positions)
    print("Modes:", env.agent_modes)
    print("Rewards:", rewards)
    
    # What if 2 agents try to move into the SAME square, but neither passes?
    # agent 0 is 2,1, moves right (3) -> 3,1
    # agent 1 is 2,3, moves right (3) -> 3,3
    env.step({"agent_0": 3, "agent_1": 3, "agent_2": 2})
    print("Positions after move 3:", env.agent_positions)
    
    # Now agent 0 is at 3,1. Moves down (1) -> 3,2
    # agent 1 is at 3,3. Moves up (0) -> 3,2
    # They should bounce.
    obs, rewards, term, trunc, info = env.step({"agent_0": 1, "agent_1": 0, "agent_2": 4})
    print("Positions after bounce attempt:", env.agent_positions)
    print("Modes:", env.agent_modes)
    print("Rewards:", rewards)
    

if __name__ == "__main__":
    test_env()
