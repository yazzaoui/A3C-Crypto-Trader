
initial_price = 400.
rate = 1.10
block_reward_rate = 0.94
GPU_price = float(input("Prix GPU: "))
GPU_Conso = float(input("Conso W: "))
hashrate = float(input("Hashrate H/S:  "))
price_kwh = 0.15
current_block_reward = 5.61
network_hashrate = 486.2e6
mined = 0
price = initial_price
for i in range(12):
    
    mined += (( (hashrate) * (current_block_reward) * 720 ) / (network_hashrate) ) * 30
    current_block_reward = block_reward_rate * current_block_reward
    price = price * rate
    network_hashrate = network_hashrate * rate


spending = GPU_price + (GPU_Conso * 365 * 24 * price_kwh)/1000
ca_an = mined * price

inv_direct = ((spending * price)/initial_price - spending)
print("Prix Monero Final: %.2f" % price)
print("Monero minés: %.2f" % mined)
print("Chiffre d'affaire : %.2f" % (ca_an))
print("Depenses: %.2f" % spending)
print("Profit: %.2f" % (ca_an - spending))
print("Profit si investissement direct: %.2f" % (inv_direct))
print("Rentabilite: %.2f pc" % ((ca_an - spending) * 100/spending))