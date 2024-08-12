# An Empirically Grounded Analytics (EGA) Approach to Hog Farm Finishing Stage Management: Deep Reinforcement Learning as Decision Support and Managerial Learning Tool
## About
This depository is the companion code for the paper - An Empirically Grounded Analytics (EGA) Approach to Hog Farm Finishing Stage Management: Deep Reinforcement Learning as Decision Support and Managerial Learning Tool.

The paper is available at SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4617964

## Usage
Run myddpg_per.py for training and testing. To set a simulation environment for training and testing, modify the parameter in the code accordingly (TradingEnv(...)).

For hyperparameter tuning, run run.py.

To use each file, please refer to the detailed comments and documentation within.

## Credits
- The code structure is adapted from @xiaochus' github project Deep-Reinforcement-Learning-Practice (https://github.com/xiaochus/Deep-Reinforcement-Learning-Practice).
- The prioritized experience replay buffer implementation is taken from OpenAI Baselines (https://github.com/openai/baselines).
