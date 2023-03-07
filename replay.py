import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import random, datetime
from pathlib import Path
import matplotlib.pyplot as plt

from metrics import MetricLogger
from agent import Agent
from env import env


trade_env = env(learning_length=3, batch_iter=0, file_name="out_BTCUSDT.csv")  # l_l:학습할 길이(총 시간), b_i:학습할 길이로 자른 배치의 순서 선택

save_dir = Path('results') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

save_dir.mkdir(parents=True)

checkpoint = Path('checkpoints/2022-07-25T19-32-18/Trade_net_130.chkpt')

#state_data, price_data = next(iter(trade_env.data))

trade_agent = Agent(state_size=3, save_dir=save_dir, checkpoint=checkpoint, batch_num=32)
trade_agent.exploration_rate = trade_agent.exploration_rate_min
logger = MetricLogger(save_dir)

episodes = 11


for e in range(episodes):

    state = trade_env.reset()
    L_num = 0.1
    prev_num = 0
    trade_agent.curr_step = 0

    # Play the game!
    while True:

        # 4. Run agent on the state
        q_value, action = trade_agent.act(state)  # 여기서 curr_step이 1올라감

        # 5. Agent performs action
        next_state, reward, total_profit, prev_num, done = trade_env.step(action, q_value, L_num, prev_num, trade_agent.curr_step)

        # 6. Remember
        trade_agent.cache(state, next_state, action, reward, done)

        # 8. Logging
        logger.log_step(reward.detach().cpu().numpy(), None, None)


        # 9. Update state
        state = next_state

        # 10. Check if end of game
        if done:
            break

    logger.log_episode()
    logger.log_num(prev_num, action, reward)


    if e % 5 == 0:
        logger.record(
            episode=e,
            epsilon=trade_agent.exploration_rate,
            step=trade_agent.curr_step
        )


'''
a=trade_env.price_data[:2500]
a=a.detach().cpu().numpy()

fig = plt.figure(figsize=(20, 8))
plt.plot(a)
save_dir = save_dir / "price.png"
plt.savefig(save_dir)
plt.close(fig)
'''