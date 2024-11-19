# CS-4710-Semester-Project

Authors: Ethan Kacena-Merrell, Jalen Wang, Jerry Gu, Pranav Arora, Rithwik Raman, Timofey Maslyukov, and Jaeson Martin


Instructions to Run the Code
Install Required Libraries:
pip install pygame numpy pandas tensorflow matplotlib

Download the Code:
Save each code block into separate .py files as indicated:

snake_game.py
snake_game_data_collection.py
snake_game_ai.py
snake_rl.py

Run the Basic Snake Game:
python snake_game.py
Use the arrow keys to control the snake.

Collect Data for AI Training:
python snake_game_data_collection.py
The game will run autonomously, and data will be saved to snake_game_data.csv.

Run the Snake Game with Pathfinding AI:
python snake_game_ai.py
The snake will navigate towards the pellet using the A* algorithm. You can switch to BFS by uncommenting the relevant line in the code.

Train the Reinforcement Learning Agent:
python snake_rl.py
The agent will begin training over 1000 episodes. Training progress and metrics will be displayed.

Visualize Training Results:
After training, plots of scores, survival times, losses, and epsilon values will be displayed.

Test the Trained Agent:
To test the agent without further training, modify snake_rl.py to load the saved model and run test episodes.

agent.model = tf.keras.models.load_model('snake_dqn_model.h5')
agent.update_target_model()


agent.epsilon = 0

def test_agent(agent, game, episodes=5):
    for e in range(episodes):
        state = game.reset()
        done = False
        score = 0
        steps = 0
        while not done:
            game.render()
            action = agent.act(state)
            state, reward, done, score = game.step(action)
            steps += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        print(f"Test Episode {e+1}/{episodes}, Score: {score}, Steps: {steps}")
        pygame.time.delay(1000)  # Pause between episodes

    pygame.quit()

test_agent(agent, game)


Run the script again to observe the agent's performance.

