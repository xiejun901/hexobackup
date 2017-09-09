title: 强化学习-DP, MC, TD
date: 2017-08-27 20:11:23
tags: [强化学习, 机器学习, DP, MC, TD]
tags: machine laearning
---

上一份笔记记录了强化学习的一些基本概念，其中提到了最优策略，本次笔记记录的就是求解在指定策略下的状态值函数(动作值函数)和最优策略的方法。 

在众多求解最优策略的方法中，求解过程都可以划分成两个部分，策略求值(policy evaluation)和策略提升(policy improvement). 其中策略求值的方法有很多种，主要包括DP，MC, TD 等，策略提升一般使用贪心，$\epsilon-greedy$.

![](http://7q5fny.com1.z0.glb.clouddn.com/rl1.png)

![](http://7q5fny.com1.z0.glb.clouddn.com/rl2.png)

如上图，途途中的$q_\pi$同事可以替换为$v_\pi$，策略求值可以计算到再指定策略下状态值函数(动作值函数)收敛也可以是仅仅迭代一次或者多次之后就进行策略提升。

### DP

####  prediction

动态规划的求值利用的bellman公式进行计算, 是一个迭代过程。通过反复的迭代，可以收敛至$v_\pi$

$$
v_{k+1}(s)=\sum_{a \in A}{\pi(a|s)(R_s^a + \gamma \sum_{s \in S^\prime}{P_{ss^\prime} v_k(s^\prime)})}
$$

#### control

可以反复迭代evaluation过程收敛至$v_\pi$之后进行policy improvement. 新的策略更新为 对于指定的状态s, 选择后继$v(s^\prime)$最大的状态的action, 这种做法成为策略迭代，在迭代的过程中有显示的策略存在

同时, 也可以进行一次计算就更新策略,这种做法没有显示的策略存在，称为值迭代。计算公式如下:

$$
v_{k+1}(s)=max_{a \in A}(R_s^a+\gamma \sum_{s^\prime \in s}{P_{ss^\prime}^a{v_k(s^\prime)}})
$$

### MC

#### prediction

上面的动态规划里面很核心的东西是需要知道状态之间的转移概率，这个在很多实际情况下都是完全不能的，因此希望能通过采样来进行需要，能够从经验中来进行学习。

对于基于 $\pi$ 产生的很多个episodes, 我们可以用多个episodes的reture的均值来作为状态的$v$

S1,A1,R1,S2,A2,R2,S3,A3,R3 ... 

$$
v_\pi(s)=E[G_t|S_t=s]
$$

实际上，我们可以使用如下公式进行计算

$$
v(S_t) = v(S_t) + \alpha (G_t - v(S_t))
$$

同时，也可以使用q函数来进行计算

$$
q(s,a) = q(s,a) + \alpha (G_t - q(s, a))
$$

#### control

同样可以使用计策略求值，策略提升的方式来求最佳策略，但是我们需要做 model free 的最优策略计算，可以看看下面公式

$$
\pi_\*(s)=argmax_{a \in A}(R_s^a + \sum_{s' \in S}{P_{ss^\prime}^a V(s^\prime)})
$$

$$
\pi_\*(s)=argmax_{a \in A}Q(s,a)
$$

显然，没办法使用 V 来做策略提升， 因为使用V来做策略提升就必须知道下一步应该会走到哪些$s^\prime$,这是模型的一部分。
因此只能使用 Q 来进行 control

使用 monte-carlo 来做control事下面这样的过程

- 使用策略 $\pi$ 产出一个episode, S1,A1,R1,S2,A2,R2........
- 针对每个 t 计算 出 $G_t$ 
- 针对每个状态，统计出计算次数和 return 的期望

$$
N(S_t, A_t) = N(S_t, A_t) + 1
$$

$$
Q(S_t, A_t) = Q(S_t, A_t) + \frac{1}{N(S_t, A_t)}(G_t - Q(S_t, A_t))
$$

- 根据当前的Q值使用 $\epsilon-greedy$ 改善策略 

### TD


