import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import random, datetime
from pathlib import Path
import matplotlib.pyplot as plt

from metrics import MetricLogger
from agent import Agent
from env import env

# Initialize environment
trade_env = env(learning_length=2602, batch_iter=0, file_name="out_BTCUSDT_220705.csv") # l_l:학습할 길이(총 시간), b_i:학습할 길이로 자른 배치의 순서 선택

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = None  # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')

#state_size, _ = trade_env.coindata[1] #하나 부를땐 이거
#state_data, price_data = next(iter(trade_env.data)) #데이터 로더에 넣어서 부를땐??

trade_agent = Agent(state_size=3, save_dir=save_dir, batch_num=32, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes = 201

### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    state = trade_env.reset()
    L_num = 100
    prev_num = 0
    trade_agent.curr_step = 0

    # Play the game!
    while True:

        # 4. Run agent on the state
        q_value, action = trade_agent.act(state) #여기서 curr_step이 1올라감

        # 5. Agent performs action
        next_state, reward, total_profit, prev_num, done = trade_env.step(action, q_value, L_num, prev_num, trade_agent.curr_step)

        # 6. Remember
        trade_agent.cache(state, next_state, action, reward, done)

        # 7. Learn
        q, loss = trade_agent.learn()

        # 8. Logging
        logger.log_step(reward.detach().cpu().numpy(), loss, q)



        # 9. Update state
        state = next_state

        # 10. Check if end of game
        if done:
            break

    logger.log_episode()

    if e % 10 == 0:
        logger.record(
            episode=e,
            epsilon=trade_agent.exploration_rate,
            step=trade_agent.curr_step
        )

