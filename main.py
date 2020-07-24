from preprocessing.preprocessing import preprocessing_data
import time


def main():
    preprocessing_data()

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Execution time \n--- %s s ---" % (time.time() - start_time))