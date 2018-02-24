# A3C-Crypto-Trader
Deep Reinforcement Learning A3C Crypto Trading Bot

The trading logic is in ai.py. 
The market data scraper in scrap.py , currently only using bitmex data.

The bot was succesfully working with simple hand created market datawith a 3-layer FNN, learning profitable strategies under 30 min.
On real world data, the learning time is way too high.

The RL architecture is based on google deepmind's A3C. The trading rules (BUY/SELL/HOLD) were kept simples for maximum generalization.
Activation function is Tanh, ReLu was found not effective at all after much experimentation.

It's currently performing a grid-search to find out the best hyper-parameters, simulations results are VERY sensitive to those values.

TODO : 
- [ ] Try out an LSTM network.
- [ ] Implement generative algorithm to find optimal hyperparameters.
- [ ] Try out RAINBOW.



Created by Youssef Azzaoui based on Google Deepmind's work.
If any question, please contact me y@azzaoui.fr
