import subprocess
import model
import multiprocessing

def worker(config_file_path):
    model.main(config_file_path)

if __name__ == '__main__':
    with open("LIST", 'r') as in_file:
        pool = multiprocessing.Pool(processes=10)
        for row in in_file:
            pool.apply_async(worker, args=(row.split()[-1],))
        pool.close()
        pool.join()