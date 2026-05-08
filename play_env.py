import time
import pygame
from swarm_env import RationalSwarmForagingEnv

def main():
    # Initialize environment with human render mode
    env = RationalSwarmForagingEnv(n_agents=10, grid_size=12, num_food=12, render_mode="human")
    
    obs, info = env.reset()
    env.render()
    
    print("Pygame viewer started.")
    print("Press SPACE to step the environment (agents will take random actions).")
    print("Press ESC or close the window to quit.")
    
    running = True
    auto_play = False
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Take random actions and step
                    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
                    obs, rewards, term, trunc, info = env.step(actions)
                    env.render()
                elif event.key == pygame.K_a:
                    auto_play = not auto_play
                    print(f"Auto-play toggled: {auto_play}")
                    
        if auto_play:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, term, trunc, info = env.step(actions)
            env.render()
            time.sleep(0.5) # slow down enough to see
            
    env.close()

if __name__ == "__main__":
    main()
