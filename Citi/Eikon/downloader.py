import time

from uti import DataLoader, Logger, timeit
from Eikon.settings import myThread, chunks

DL = DataLoader()
logger = Logger()
#################### Price, Market Cap data daily update ######################

@timeit
def Eikon_update_price_enhanced(tickers, threadcount=8):
    # threadcount = 4 # threading.activeCount()
    n = len(tickers) // threadcount
    threads = []
    print(n)
    if n > 0:
        tickers_chunk = list(chunks(tickers, n))
    else:
        tickers_chunk = [tickers]
        threadcount = 1


    for i in range(threadcount):

        tickers_group = list(tickers_chunk[i])
        thread = myThread(i+1, tickers=tickers_group)
        threads.append(thread)
        thread.start()
        time.sleep(0.0001)

    for t in threads:
        t.join()

    # Check anomalies
    # anomalies_df = DL.loadDB(f'Log/Anomalies_MC_{NOW_STR}.csv')
    # anomalies_df = DL.loadLog()
    # if len(anomalies_df) > 0:
    #     logger.info('Anomalies tickers: ')
    #     logger.info(anomalies_df)
    # else:
    #     logger.info("All tickers completed without error.")

    print("Exiting Main Thread")