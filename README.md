# CS-4710-Semester-Project

Authors: Ethan Kacena-Merrell, Jalen Wang, Jerry Gu, Pranav Arora, Rithwik Raman, Timofey Maslyukov, and Jaeson Martin

Instructions to Run the Code Install Required Libraries: pip install pygame numpy pandas tensorflow matplotlib

Download the Code, including the models in the Models file.

Run the Basic Snake Game: python snake_game.py Use the arrow keys to control the snake.

Collect Data for AI Training: python snake_game_data_collection.py The game will run autonomously, and data will be saved to snake_game_data.csv.

Run the Snake Game with Pathfinding AI: python snake_pathfinding_ai.py The snake will navigate towards the pellet using the A* algorithm. You can switch to BFS by uncommenting the relevant line in the code.

Train and test the Reinforcement Learning Agents (training is usually around 1000 episodes):
- python snake_game_naive_winning_strategy.py will play the naive strategy
- python snake_tabular.py will train and test the Tabular Q-Learning model
- python snake_rl_dqn.py will train the DQN model (2 hidden layers), snake_dqn_model.h5 is the trained model, and snake_test_for_rl_dqn.py will test the model
- python snake_rl_more_hidden_layers.py will train the DQN model with 3 hidden layers and more_layers_snake_model.h5 is the trained model
- python snake_rl_ddqn.py will train the double DQN model, ddqn_snake_model.h5 is the trained model, and snake_test_for_rl_ddqn.py will test the model
- python snake_rl_dueling.py will train the dueling DQN model. It will generate a model of the format [# of epochs]_epoch_snake_model, whihc can be run using snake_test_rl_dueling.py

Visualize Training Results: After training, plots of scores, survival times, losses, and epsilon values will be displayed.

Test the Trained Agent: To test the agent without further training, modify the associated test file to load the saved model and run test episodes.

Run the script again to observe the agent's performance.
