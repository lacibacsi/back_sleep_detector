
import cmd
import back_sleep


# simple demo file for showing the model to David and Samuel

class BackTestDemo(cmd.Cmd):

    def __init__(self):
        super(BackTestDemo, self).__init__()

        self._backmodel = back_sleep.BackSleepDetector()

    def do_help(self, line):
        print("use: train for training, predict <filename> for prediction on an image")

    def do_train(self, line):
        print("starting training on default dataset")
        self._backmodel.train()

    def do_predict(self, line):
        fname = '../dataset/predict/snapshot-00007.jpg'
        if line:
            fname = line

        print(f'calling prediction on {fname}')
        result = self._backmodel.predict(fname)

        print(result)
        print(f'the model predicted that the person is {"not " if not result else ""}laying on their back')

    def do_quit(self, line):
        return True

    def do_EOF(self, line):
        return True


if __name__ == '__main__':
    BackTestDemo().cmdloop()