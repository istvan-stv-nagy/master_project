from run.runnable import Runnable

if __name__ == '__main__':
    for i in range(0, 30):
        runnable = Runnable()
        runnable.run(i)
