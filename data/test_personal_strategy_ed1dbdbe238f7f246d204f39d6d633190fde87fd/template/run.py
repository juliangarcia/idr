import subprocess
import model
import multiprocessing

def worker(config_file_path):
    model.main(config_file_path)

if __name__ == '__main__':
    with open("LIST", 'r') as in_file:
        pool = multiprocessing.Pool(processes=10)
        for row in in_file:
            # subprocess.run(row.split())
            pool.apply_async(worker, args=(row.split()[-1],))
            # p = multiprocessing.Process(target=worker, args=(row.split()[-1],))
            # p.start()
        pool.close()
        pool.join()
