PRICE_BTC_START = 16000
PRICE_BTC_END = 16000

DIFFICULTY_START = 1.71e12
DIFFICULTY_RATE = 1.17 # PESSIMISTE 1.25

#MATOS
HASHRATE = 14e12
PRIX_MINER = 3000
CONSO_KW = 1.4

COST_KWH = 0.15
MOIS = 12
print("Resultats Simulation - 12 mois")
print("Prix BTC début: %d$ | fin: %d$" % (PRICE_BTC_START,PRICE_BTC_END))
print("Prix Miner: %d$ - Conso : %.2fKw - Prix KwH: %.2f$" %(PRIX_MINER,CONSO_KW,COST_KWH))
difficulty = DIFFICULTY_START
total_ca = 0
btc_total = 0
for i in range(MOIS):
    price_btc = i * (PRICE_BTC_END - PRICE_BTC_START)/11 + PRICE_BTC_START
    block_per_time = HASHRATE /(difficulty * 2**32)
    block_per_month = block_per_time * 30 * 24 * 60 * 60
    btc_total += block_per_month * 12.5
    money_this_month = block_per_month * price_btc * 12.5
    total_ca += money_this_month
    print("MOIS %d - chiffre d'affaire : %.2f $" % (i+1,money_this_month))
    difficulty = difficulty * DIFFICULTY_RATE

depenses = PRIX_MINER + COST_KWH * MOIS * 30 * 24 * CONSO_KW
profit = total_ca - depenses
print("\nChiffre d'affaire : %.2f$" % total_ca)
print("Depenses totales: %.2f$" % depenses )
print("Profit: %.2f$" %profit)
print("Profit2: %.2f$" %(btc_total * PRICE_BTC_END - depenses))
print("BTC miné: %.2f" %btc_total)
print("Valeur si BTC conservés: %.2f$" %(btc_total * PRICE_BTC_END) )
profit_dir = (depenses * PRICE_BTC_END) / PRICE_BTC_START 
print("Profit si achat BTC de %.2f$ : %.2f$" % (depenses,(profit_dir-depenses)))
print("\n*****************************\n")