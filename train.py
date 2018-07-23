import pygame
import tensorflow as tf
import pong_game
import numpy as np

restore = False
mainloop = True
gamma = 0.96


input_frame = tf.placeholder(tf.float32, [380*400, None], name='input_frame')
# label = tf.placeholder(tf.float32, [None, 3], name='label')

W1 = tf.get_variable(name='w1', shape=[200, 380*400],initializer=tf.contrib.layers.xavier_initializer(seed=2))
b1 = tf.get_variable(name='b1', shape=[200, 1], initializer=tf.zeros_initializer())
W2 = tf.get_variable(name='w2', shape=[3, 200],initializer=tf.contrib.layers.xavier_initializer(seed=2))
b2 = tf.get_variable(name='b2', shape=[3, 1], initializer=tf.zeros_initializer())


def build_model():
    L_1 =tf.nn.relu(tf.add(tf.matmul(W1, input_frame), b1))
    L_2 =tf.nn.relu(tf.add(tf.matmul(W2, L_1), b2))

    Y = tf.nn.softmax(L_2, axis=0)
    return Y

def discont_reward(reward_, discont_factor_ = 0.8):
    reward_ = np.array(reward_)
    episode_length = reward_.size
    disconted_reward = np.zeros(reward_.shape)
    for t in reversed(range(0, episode_length)):
        disconted_reward[t] = reward_[episode_length - 1] * discont_factor_ ** (episode_length - 1 - t)
    return disconted_reward


def discount_rewards(r):
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, discounted_r.size)):
    if r[t] != 0: running_add = 0
    #https://github.com/hunkim/ReinforcementZeroToAll/issues/1
    running_add = running_add * gamma + r[t]

    discounted_r[t] = running_add
  return discounted_r


my_pong = pong_game.pong()

sess = tf.Session()

Y = build_model()
sample_op = tf.multinomial(logits=tf.reshape(Y, (1, 3)), num_samples=1)
Y_action = sample_op - 1

frame_count = 0
last_frame = []
episode_menory = []
episode = 0

saver = tf.train.Saver()
global_step = tf.train.get_or_create_global_step()
init = tf.global_variables_initializer()


if restore:
    saver.restore(sess, tf.train.latest_checkpoint('./check_point'))
    print('Check point restored!')
else:
    sess.run(init)

while mainloop:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            mainloop = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                mainloop = False

    clock = pygame.time.Clock()
    clock.tick(100)

    next_frame = my_pong.get_next_frame().T[pong_game.SCOREBOARD_HEIGHT + 6 : pong_game.WINDOW_HEIGHT, 0 + pong_game.PADDLE_THICKNESS:pong_game.WINDOW_WIDTH - pong_game.PADDLE_THICKNESS].reshape(380*400, 1)
    frame_count += 1

    if frame_count > 1:

        observation_ = next_frame - last_frame
        action_ = sess.run(Y_action, feed_dict={
            input_frame: observation_})
        # action_ = np.squeeze(action_)
        # print(action_)
        done_, reward_ = my_pong.paddle_2_control(action_)
        episode_menory.append((observation_, action_, float(reward_)))
        # print(done_, reward_)

        if done_:
            obs, lab, rwd = zip(*episode_menory)
            prwd = discount_rewards(rwd)
            prwd -= np.mean(prwd)
            prwd /= np.std(prwd)

            obs_reshape = np.squeeze(obs).T

            lab_one_hot = tf.one_hot(np.squeeze(lab)+1, 3, axis=0)

            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=lab_one_hot, logits=Y)

            cost = cross_entropy * prwd

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            train_op = optimizer.minimize(cost)

            _, cost_ = sess.run([train_op, cost], feed_dict={input_frame: obs_reshape})

            episode_menory = []

            print("Episode %d finish! Cost = %f" % (episode, np.sum(cost_)))

            episode += 1

            if episode % 500 == 0:
                saver.save(sess, './check_point/model_iter', global_step=episode)



        # print(done_, reward_)

    last_frame = next_frame





