import rlgallery
from rlgallery import RUNNERS, ENVS

def main():
    run = RUNNERS.build({'type': "Runner"})
    env = ENVS.build({"type": "Envs"}).gym_env

    if run.algo not in ["PPOContinuous","DDPG", "SACContinuous"] and env.is_continus:
        print(f"{run.algo} for continuous env has not been implemented yet!")
        return
    if run.algo in ["VPG", "REINFORCE", "ActorCritic", "TRPO", "PPO", "PPOContinuous"]:
        run.train_on_policy_agent()
    else:
        run.train_off_policy_agent()

if __name__ == "__main__":
    main()