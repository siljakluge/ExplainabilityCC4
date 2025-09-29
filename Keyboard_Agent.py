from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent, KeyboardAgent

steps = 1000
sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent, # use a stand-in agent that you will overwrite the actions of
                                green_agent_class=EnterpriseGreenAgent, 
                                red_agent_class=FiniteStateRedAgent,
                                steps=steps)
cyborg = CybORG(scenario_generator=sg, seed=1234)

# Create the keyboard agent
agent = KeyboardAgent('blue_agent_0')
# Reset the environment
results = cyborg.reset()

for i in range(100):
    # Get the action and observation space
    obs = results.observation
    action_space = cyborg.get_action_space('blue_agent_0')

    # Prompt the keyboard agent to ask yoy for the action to take
    action = agent.get_action(obs, action_space)
    # Take a step using that action
    results = cyborg.step(agent='blue_agent_0', action=action)